Clear[muB, muK, muQ, temp, a];
mu = {muB/3 + 2*muQ/3, muB/3 + muK/2 - muQ/3,
      muB/3 - muK/2 - muQ/3};
density[x_] := x^3/Pi^2 + x*temp^2;
densities = density /@ mu;
nB = Total[densities]/3;
nK = (densities[[2]] - densities[[3]])/2;
aSimple = Factor[nK/nB /. muQ -> 0];
neutrality = Factor[6*Pi^2*(2*densities[[1]]-densities[[2]]-densities[[3]])];
inverseNeutral = muB*((1+a)^(1/3)-(1-a)^(1/3)) /
    (1+(1+a)^(1/3)+(1-a)^(1/3));
{aSimple, neutrality,
 NSolve[(aSimple /. {muB -> 1200, temp -> 0}) == 1 && 0 < muK < 800,
        muK, Reals],
 N[inverseNeutral /. {muB -> 1200, a -> 1}, 12],
 Factor[D[aSimple, temp]]}
