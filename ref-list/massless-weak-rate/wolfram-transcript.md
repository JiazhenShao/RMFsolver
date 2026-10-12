# Wolfram verification transcript

Executed through the connected Wolfram Language evaluator on 2026-10-07.
The remote input first embeds `spin-trace.wl`; local execution loads that file with `Get`.
Computational symbols `u,d,s` mean flavor chemical potentials, not hydrodynamic velocities.
The physical branch is `0 <= s < u < d`; all branch integrals are supplied explicitly.

## Complete evaluated input

```wolfram
spinCheck = (* Four-spin-summed current-current trace in the (+---) metric.
   aa=p1, bb=pd, cc=p2, dd=ps.  No on-shell assumption is needed.
   The Hamiltonian coupling is (GF/Sqrt[2])*Sin[thetaC]*Cos[thetaC]
   multiplying two currents with (1-gamma5), not (1-gamma5)/2.

   Independently evaluated with Wolfram MCP on 2026-10-07.
   Expected result:
   <|"TraceResidual" -> 0, "FourSpinSumCoefficient" -> 128,
     "ColorSum" -> 9, "SpinColorCoefficient" -> 1152,
     "MadsenCombinedCoefficient" -> 1152|>
*)

Module[{id, zz, gamma, gamma5, left, metric, slash, mdot,
        aa, bb, cc, dd, a, b, c, d, trace, residual},
  id = IdentityMatrix[2];
  zz = ConstantArray[0, {2, 2}];
  gamma = Join[
    {ArrayFlatten[{{id, zz}, {zz, -id}}]},
    Table[ArrayFlatten[{{zz, PauliMatrix[j]}, {-PauliMatrix[j], zz}}],
          {j, 1, 3}]
  ];
  gamma5 = I Dot @@ gamma;
  left = IdentityMatrix[4] - gamma5;
  metric = {1, -1, -1, -1};
  slash[p_] := Sum[metric[[j]] p[[j]] gamma[[j]], {j, 4}];
  mdot[p_, q_] := Sum[metric[[j]] p[[j]] q[[j]], {j, 4}];
  aa = Array[a, 4];
  bb = Array[b, 4];
  cc = Array[c, 4];
  dd = Array[d, 4];
  trace = Sum[
    metric[[j]] metric[[k]]
    Tr[slash[cc].gamma[[j]].left.slash[bb].gamma[[k]].left]
    Tr[slash[dd].gamma[[j]].left.slash[aa].gamma[[k]].left],
    {j, 4}, {k, 4}
  ];
  residual = Expand[trace - 256 mdot[aa, bb] mdot[cc, dd]];
  If[residual =!= 0, Print["Spin trace verification failed: ", residual]; Abort[]];
  <|"TraceResidual" -> residual, "FourSpinSumCoefficient" -> 256/2,
    "ColorSum" -> 3^2, "SpinColorCoefficient" -> (256/2)*3^2,
    "MadsenCombinedCoefficient" -> (36/2)*64|>
];
(* Massless d+u -> s+u rate; natural units and metric +---.
   Computational u,d,s denote mu_u,mu_d,mu_s, not velocities.
   Physical branch assumptions: 0 <= s < u < d; temperature tends to 0+.
   Polynomial integrations use explicitly specified branches, not implicit
   Integrate assumptions. All residuals are exact, not numerical tests.
   Run: wolframscript -file verification.wl
   The remote verifier supplies spinCheck from spin-trace.wl before this file. *)

If[!ValueQ[spinCheck],
  spinCheck = Get[FileNameJoin[{DirectoryName[$InputFileName], "spin-trace.wl"}]]];
Clear[u,d,s,x,y,en,m,z,a,t,ma,mb,p,cu,cd,cs];
kernel[en_,m_] := 32 en^2 m^3/3 - 16 en m^4 + 32 m^5/5;
weight[en_,m_] := 32 m^2 (en-m)^2;
energyIntegral = Integrate[kernel[en,m],{en,u+y,x+d}];
f1 = Integrate[energyIntegral /. m->x,{y,s,x+d-u}];
f2 = Integrate[energyIntegral /. m->y,{y,s,x}];
f3 = Integrate[energyIntegral /. m->x,{y,x,x+d-u}];
common = Integrate[f2+f3,{x,s,u}];
coldLarge = Expand[Integrate[f1,{x,0,s}]+common];
coldSmall = Expand[Integrate[f1,{x,u+s-d,s}]+common];
polynomial = (u^4(210d^4-112d^3 u-84d^2 u^2-32d u^3+65u^4)
 +s^4(168d^3 s-280d^3 u-168d^2 s^2+504d^2 s u-420d^2 u^2
 +80d s^3-336d s^2 u+504d s u^2-280d u^3-18s^4+64s^3 u
 -196s^2 u^2+168s u^3+210u^4))/315;
coldCorrection=(u+s-d)^6(15d^2+10d(s+u)+3(s+u)^2)/315;

upper = Integrate[weight[u+d,m](d-s),{m,0,s}]+
       Integrate[weight[u+d,m](d-m),{m,s,u}];
lowerSmall = Integrate[weight[u+s,m](d-s),{m,0,u+s-d}]+
       Integrate[weight[u+s,m](u-m),{m,u+s-d,s}];
lowerLarge = Integrate[weight[u+s,m](u-m),{m,0,s}];
edge1 = Integrate[weight[en,u](en-2u),{en,2u,u+d}];
edgeDsmall=Integrate[weight[d+m,m](d+m-u-s),{m,u+s-d,s}]+
       Integrate[weight[d+m,m](d-u),{m,s,u}];
edgeDlarge=Integrate[weight[d+m,m](d+m-u-s),{m,0,s}]+
       Integrate[weight[d+m,m](d-u),{m,s,u}];
edge2=Integrate[weight[u+m,m](u-m),{m,s,u}];
edgeS=Integrate[weight[en,s](u-s),{en,u+s,d+s}]+
       Integrate[weight[en,s](u+d-en),{en,d+s,u+d}];
jLarge=Expand[2upper+2lowerLarge+edge1-edgeDlarge+edge2-edgeS];
jSmall=Expand[2upper+2lowerSmall+edge1-edgeDsmall+edge2-edgeS];
jPolynomial=8/15*(20d^3 s^3-35d^2 s^4+32d s^5-11s^6
 +u(-20d^3 s^2+60d^2 s^3-80d s^4+24s^5)
 +u^2(15d^4-30d^2 s^2+60d s^3-35s^4)
 +u^3(-20d s^2+20s^3)+u^4(10d^2+15s^2)-8d u^5+8u^6);
jCorrection=-8/15*(u+s-d)^4*(9d^2+4d s+2s^2+4d u+4s u+2u^2);

primitive=(-t Log[1+Exp[(ma-p)/t]]+t Log[1+Exp[(p-en+mb)/t]]-p)/
          (1-Exp[(en-ma-mb)/t]);
fermiProduct=1/((1+Exp[(p-ma)/t])*(1+Exp[(en-p-mb)/t]));
primitiveResidual=FullSimplify[D[primitive,p]-fermiProduct,
  Assumptions->t>0 && Element[{ma,mb,p,en},Reals] && en!=ma+mb];
removableResidual=FullSimplify[
  D[-t/(1+Exp[(p-ma)/t]),p]-(fermiProduct/.en->ma+mb),
  Assumptions->t>0 && Element[{ma,mb,p},Reals]];

neutralShifts=First[Solve[{cu+cd+cs==0,cd-cs==0,
  2(3u^2 cu/Pi^2+u)-(3d^2 cd/Pi^2+d)-(3s^2 cs/Pi^2+s)==0},
  {cu,cd,cs}]];
shiftExpected=Pi^2(2u-d-s)/(3(4u^2+d^2+s^2));
Clear[n1,nd,n2,ns];
imbalance=n1+nd-n2-ns;
simplexThermal=Simplify[Pi^2/6 Total[Table[D[imbalance^3,{v,2}],
  {v,{n1,nd,n2,ns}}]]];

checks=<|
 "SpinTrace"->spinCheck,
 "Prefactor"->Simplify[(18*64/4)*(2Pi)^4/((2Pi)^12*2^4)*16Pi^3],
 "AngularKernelResidual"->Expand[Integrate[(en^2-z^2)^2,{z,en-2m,en}]-kernel[en,m]],
 "PairPrimitiveResidual"->primitiveResidual,
 "RemovableLimitResidual"->removableResidual,
 "ColdPolynomialResiduals"->{Expand[coldLarge-polynomial],Expand[coldSmall-polynomial+coldCorrection]},
 "ColdBranchDerivatives0Through6"->Table[Factor[D[coldLarge-coldSmall,{d,n}]/.d->u+s],{n,0,6}],
 "ColdHomogeneity"->Expand[coldLarge/.{u->a u,d->a d,s->a s}]-a^8 coldLarge//Expand,
 "ThermalPolynomialResiduals"->{Expand[jLarge-jPolynomial],Expand[jSmall-jLarge-jCorrection]},
 "ThermalBranchDerivatives0Through3"->Table[Factor[D[jSmall-jLarge,{d,n}]/.d->u+s],{n,0,3}],
 "ThermalHomogeneity"->Expand[(jLarge/.{u->a u,d->a d,s->a s})-a^6 jLarge],
 "ThermalStrangeZero"->Factor[jLarge/.s->0],
 "EqualFlavorColdLeading"->Limit[Cancel[(coldSmall/.{d->u+a,s->u-a})/(2a)^3],a->0],
 "EqualFlavorThermalLeading"->Limit[Cancel[(jSmall/.{d->u+a,s->u-a})/(2a)],a->0],
 "LogisticNormalization"->Integrate[1/(4Cosh[z/2]^2),{z,-Infinity,Infinity}],
 "LogisticVariance"->Integrate[z^2/(4Cosh[z/2]^2),{z,-Infinity,Infinity}],
 "CubicBoseMoment"->Integrate[z^2/(Exp[z]-1),{z,0,Infinity}],
 "LogisticDifferenceThirdMoment"->2Sum[n^2 Factorial[4]/n^5-2n Factorial[3]/n^4,{n,1,Infinity}],
 "CubicWeight"->Factor[Integrate[32u^2 z^2,{z,s,d}]],
 "NeutralShiftResiduals"->Simplify[{cu+2shiftExpected,cd-shiftExpected,cs-shiftExpected}/.neutralShifts],
 "SimplexThermal"->simplexThermal,
 "UnequalEquilibriumAbove"->Factor[15/(256u^5) Integrate[((u+d)^2-z^2)^2,{z,d-u,d+u}]],
 "UnequalEquilibriumBelow"->Factor[15/(256u^5) Integrate[((u+d)^2-z^2)^2,{z,u-d,d+u}]]
|>;
Print[InputForm[checks]];
residualKeys={"AngularKernelResidual","PairPrimitiveResidual","RemovableLimitResidual",
 "ColdPolynomialResiduals","ColdHomogeneity","ThermalPolynomialResiduals",
 "ThermalHomogeneity","NeutralShiftResiduals"};
If[!AllTrue[Flatten[Lookup[checks,residualKeys]],TrueQ[#==0]&],
 Print["FAILED: a symbolic residual is nonzero."];Abort[]];
checks

```

## Returned output

```text
During evaluation of In[1]:= <|"SpinTrace" -> <|"TraceResidual" -> 0, "FourSpinSumCoefficient" -> 128, "ColorSum" -> 9, "SpinColorCoefficient" -> 1152, "MadsenCombinedCoefficient" -> 1152|>, "Prefactor" -> 9/(8*Pi^5), "AngularKernelResidual" -> 0, "PairPrimitiveResidual" -> 0, "RemovableLimitResidual" -> 0, "ColdPolynomialResiduals" -> {0, 0}, "ColdBranchDerivatives0Through6" -> {0, 0, 0, 0, 0, 0, 64*(s + u)^2}, "ColdHomogeneity" -> 0, "ThermalPolynomialResiduals" -> {0, 0}, "ThermalBranchDerivatives0Through3" -> {0, 0, 0, 0}, "ThermalHomogeneity" -> 0, "ThermalStrangeZero" -> (8*u^2*(15*d^4 + 10*d^2*u^2 - 8*d*u^3 + 8*u^4))/15, "EqualFlavorColdLeading" -> (128*u^5)/45, "EqualFlavorThermalLeading" -> (1024*u^5)/15, "LogisticNormalization" -> 1, "LogisticVariance" -> Pi^2/3, "CubicBoseMoment" -> 2*Zeta[3], "LogisticDifferenceThirdMoment" -> 24*Zeta[3], "CubicWeight" -> (32*(d - s)*(d^2 + d*s + s^2)*u^2)/3, "NeutralShiftResiduals" -> {0, 0, 0}, "SimplexThermal" -> 4*(n1 - n2 + nd - ns)*Pi^2, "UnequalEquilibriumAbove" -> (10*d^2 + 5*d*u + u^2)/(16*u^2), "UnequalEquilibriumBelow" -> (d^3*(d^2 + 5*d*u + 10*u^2))/(16*u^5)|>

Out[1]= <|"SpinTrace" -> <|"TraceResidual" -> 0, "FourSpinSumCoefficient" -> 128, "ColorSum" -> 9, "SpinColorCoefficient" -> 1152, "MadsenCombinedCoefficient" -> 1152|>, "Prefactor" -> 9/(8*Pi^5), "AngularKernelResidual" -> 0, "PairPrimitiveResidual" -> 0, "RemovableLimitResidual" -> 0, "ColdPolynomialResiduals" -> {0, 0}, "ColdBranchDerivatives0Through6" -> {0, 0, 0, 0, 0, 0, 64*(s + u)^2}, "ColdHomogeneity" -> 0, "ThermalPolynomialResiduals" -> {0, 0}, "ThermalBranchDerivatives0Through3" -> {0, 0, 0, 0}, "ThermalHomogeneity" -> 0, "ThermalStrangeZero" -> (8*u^2*(15*d^4 + 10*d^2*u^2 - 8*d*u^3 + 8*u^4))/15, "EqualFlavorColdLeading" -> (128*u^5)/45, "EqualFlavorThermalLeading" -> (1024*u^5)/15, "LogisticNormalization" -> 1, "LogisticVariance" -> Pi^2/3, "CubicBoseMoment" -> 2*Zeta[3], "LogisticDifferenceThirdMoment" -> 24*Zeta[3], "CubicWeight" -> (32*(d - s)*(d^2 + d*s + s^2)*u^2)/3, "NeutralShiftResiduals" -> {0, 0, 0}, "SimplexThermal" -> 4*(n1 - n2 + nd - ns)*Pi^2, "UnequalEquilibriumAbove" -> (10*d^2 + 5*d*u + u^2)/(16*u^2), "UnequalEquilibriumBelow" -> (d^3*(d^2 + 5*d*u + 10*u^2))/(16*u^5)|>
```

All symbolic residual checks returned zero, with no evaluator warnings in this final run.
The sixth cold derivative has the stated jump; the first three thermal derivatives match.
This transcript verifies algebra and normalization within the specified collision model, not its omitted physical channels.

