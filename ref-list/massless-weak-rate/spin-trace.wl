(* Four-spin-summed current-current trace in the (+---) metric.
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
]
