(* Wolfram MCP inputs and returned results, 2026-10-16. *)

Clear[a]; ratio = ((1 + (1 + a)^(4/3) + (1 - a)^(4/3))/3)^(3/4); approxSquared = 9 (ratio - 1); {Series[approxSquared, {a, 0, 6}], N[Table[{a, Sqrt[approxSquared], 100 (Sqrt[approxSquared]/a - 1)}, {a, {0.146, 0.3, 0.5, 0.7, 0.9, 1}}], 12], FindRoot[Sqrt[approxSquared]/a == 1.01, {a, 0.5}]}

(* Returned series: a^2 + 2 a^4/27 + 61 a^6/2187 + O[a]^7.
   Returned 1% crossing: a -> 0.49554417610308704.
   Cold percentage errors at a={0.146,0.3,0.5,0.7,0.9,1}:
   {0.07955664082530678,0.3445434451802454,1.0199603745873809,
    2.234546676285043,4.532313284718215,7.05040367119470065}.
*)

Clear[muU, muD, muS, n, p]; n[m_] := m^3/Pi^2 + 20^2 m; p[m_] := m^4/(4 Pi^2) + m^2 20^2/2 + 7 Pi^2 20^4/60; Table[Module[{eqs, sol, a, muInf, approx}, eqs = {(2 n[muU] - n[muD] - n[muS])/n[420], (p[muU] + p[muD] + p[muS] + 8 Pi^2 20^4/45 - 189.1565957288247^4 - state[[1]])/(state[[1]] + 189.1565957288247^4), (muU + muD + muS + (n[muD] - n[muS])/(2 n[muU]) (muD - muS) - state[[2]])/state[[2]]}; sol = FindRoot[eqs, {{muU, 420}, {muD, 500}, {muS, 250}}, AccuracyGoal -> 12, PrecisionGoal -> 12]; a = (n[muD] - n[muS])/(2 n[muU]) /. sol; muInf = state[[3]]; approx = 3 Sqrt[(state[[2]]/muInf - 1) (muInf^2 + 3 Pi^2 20^2)/(muInf^2 + 9 Pi^2 20^2)]; {a, approx, 100 (approx/a - 1), Max[Abs[eqs /. sol]]}], {state, {{959797711.7861271, 1274.3573331968832, 1228.7398863486878}, {1903805260.4761682, 1466.992513726268, 1344.1830939805584}}}]

(* Full finite-temperature output, including precision warning:
FindRoot::lstol: The line search decreased the step size to within tolerance specified by AccuracyGoal and PrecisionGoal, but was unable to find a sufficient decrease in the merit function. You may need more than MachinePrecision digits of working precision to meet these tolerances.
General::messages: Messages were generated which may indicate errors.

Out[1]= {{0.5657807826642437, 0.5735912664197573, 1.3804787993565881, 1.0643583500093613*^-16}, {0.86552004512252, 0.9009442828495663, 4.0928269572349185, 2.426999189464384*^-16}}
   Returned constraint residuals are below 3e-16 despite the line-search warning.
*)

