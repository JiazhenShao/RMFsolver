# Fixed-threshold thermal expansion

This is an independent analytical derivation for the massless four-quark channel, at fixed chemical potentials and common temperature. It is not a finite-temperature EOS expansion or an additional antiquark-channel calculation. Numerical helpers are in `thermal_sympy.py`.

## Result and normalization

For the ordered branch `0 <= s < u < d`, with `u=mu_u`, `d=mu_d`, and `s=mu_s`, the forward phase-space integral and net rate obey

```text
P = 9*G_F^2*sin(theta_C)^2*cos(theta_C)^2/(8*pi^5)
I(T) = I0 + (pi^2/6)*J*T^2
       - (64/3)*zeta(3)*u^2*(d^3-s^3)*T^3 + O(T^4)
Gamma_K(T) = (1-exp(-(d-s)/T))*P*I(T).
```

This is a fixed-positive-imbalance asymptotic expansion. In that limit the detailed-balance multiplier differs from one by an exponentially small term. Keeping that multiplier on a truncated expansion does not turn it into a uniform approximation when `d-s` is comparable to or below `T`.

The `T^3` term is real, nonzero, and negative on the strict ordered branch. Thus retaining only the exact cold term and the quadratic thermal term has a generic `O(T^3)` remainder, not `O(T^4)`.

## Four independent thresholds

Write the incoming up/down thresholds as `A,B` and the outgoing up/strange thresholds as `C,D`. The exact finite-temperature forward integral is the convolution of the cold integral with four independent logistic threshold shifts. Each shift has zero mean and variance `pi^2*T^2/3`. Therefore

```text
J = (d_A^2 + d_B^2 + d_C^2 + d_D^2) I0(A,B,C,D)
    evaluated at A=C=u, B=d, D=s.
```

Setting `A=C` before differentiating would introduce the spurious mixed derivative `2*d_A*d_C`. Equal physical up potentials do not correlate the two independent occupation factors.

Introduce the angular minimum coordinate `m=(E-k)/2`. This turns the measure-weight product into the polynomial

```text
W(E,m) = 32*m^2*(E-m)^2
I0 = integral_0^infinity dE integral_0^(E/2) dm W(E,m)*L_in*L_out
L_in  = [min(E-m,A)-max(m,E-B)]_+
L_out = [min(E-m,E-D)-max(m,C)]_+.
```

This coordinate is useful because threshold derivatives of the interval lengths are delta distributions at linear boundaries, so all resulting integrals are elementary polynomials.

## Explicit boundary contributions

For compactness, define six one-dimensional boundary integrals below. `q=u+s-d`, `q0=max(0,q)`, and all integrations are over ordinary polynomials after the stated splits.

```text
B_hi = integral_0^s dm W(u+d,m)*(d-s)
       + integral_s^u dm W(u+d,m)*(d-m)

B_lo = integral_0^q0 dm W(u+s,m)*(u-q0)
       + integral_q0^s dm W(u+s,m)*(u-m)

E_A = integral_(2*u)^(u+d) dE W(E,u)*(E-2*u)

E_B = integral_q0^s dm W(d+m,m)*(d+m-u-s)
      + integral_s^u dm W(d+m,m)*(d-u)

E_C = integral_s^u dm W(u+m,m)*(u-m)

E_D = integral_(u+s)^(d+s) dE W(E,s)*(u-s)
      + integral_(d+s)^(u+d) dE W(E,s)*(u+d-E).

d_A^2 I0 = B_hi + E_A
d_B^2 I0 = B_hi - E_B
d_C^2 I0 = B_lo + E_C
d_D^2 I0 = B_lo - E_D

J = 2*B_hi + 2*B_lo + E_A - E_B + E_C - E_D.
```

For example, the incoming-up derivative before integrating is

```text
d_A L_in = H(A-m)*H(E-m-A)*H(B-E+A)
d_A^2 L_in = delta(A-m)*H(E-m-A)*H(B-E+A)
             - H(A-m)*delta(E-m-A)*H(B-E+A)
             + H(A-m)*H(E-m-A)*delta(B-E+A).
```

At `A=C=u`, the second term vanishes when multiplied by `L_out`; the first gives `E_A`, and the last gives `B_hi`. The analogous outgoing derivatives have the opposite first sign. Dropping the geometric boundary terms `E_A` through `E_D` does not give the correct coefficient.

## Closed polynomial

The boundary integrals simplify to the following two-branch expression:

```text
J = (8/15)*[
      20*d^3*s^3 - 35*d^2*s^4 + 32*d*s^5 - 11*s^6
      + u*(-20*d^3*s^2 + 60*d^2*s^3 - 80*d*s^4 + 24*s^5)
      + u^2*(15*d^4 - 30*d^2*s^2 + 60*d*s^3 - 35*s^4)
      + u^3*(-20*d*s^2 + 20*s^3)
      + u^4*(10*d^2 + 15*s^2) - 8*d*u^5 + 8*u^6
      - max(0,u+s-d)^4
        *(9*d^2 + 4*d*s + 2*s^2 + 4*d*u + 4*s*u + 2*u^2)
    ].
```

The only remaining branch switch is `d-s=u`. Both values and the first three derivatives of `J` match there. The polynomial is homogeneous of degree six, so `T^2*J` has the same degree eight as `I0`.

Near equal flavors the displayed polynomial suffers cancellation. The helper evaluates its exactly equivalent expansion in the dimensionless differences `(d-u)/u` and `(u-s)/u`, which remains numerically well conditioned as the imbalance tends to zero.

## Why there is a cubic thermal term

Let `x` be incoming up momentum and `z` outgoing up momentum. At zero temperature and `A<=C`, all allowed points have `x<=z`. The angular minimum is then `min(x,r)`, where `r` is outgoing strange momentum. Analytically continuing this branch to `A>C` misses the thin triangle `C<z<x<A`, in which the actual minimum is `min(z,r+z-x)`.

At `x=z=u` both changes have the same first derivative of the angular kernel:

```text
r < u: K(E,min actual)-K(E,min continued) = -32*u^2*r^2*(x-z)+O((x-z)^2)
r > u: K(E,min actual)-K(E,min continued) = -32*u^2*r^2*(x-z)+O((x-z)^2).

Q = integral_s^d 32*u^2*r^2 dr = (32/3)*u^2*(d^3-s^3)
I0(A,B,C,D) = smooth continuation - Q*(A-C)_+^3/6
             + terms of fourth or higher order about A=C.
```

The cold integral is twice continuously differentiable at the physical equal-up thresholds, making `J` unambiguous, but its third derivative has a jump. For independent standard logistic random variables `X,Y`,

```text
E|X-Y|^3 = 24*zeta(3)
E[(X-Y)_+^3] = 12*zeta(3)
```

so smoothing the cusp produces `-2*zeta(3)*Q*T^3`.

An independent direct derivation is even shorter. In the switching strip set `h=x-z>0`. Up occupations integrate to `h/(exp(h/T)-1)`, up to exponentially small lower-endpoint corrections. The kernel correction is `-Q*h`, hence

```text
correction = -Q*integral_0^infinity h^2/(exp(h/T)-1) dh
           = -2*zeta(3)*Q*T^3.
```

Both the up/up and down/strange minimum switches must be included: integrating `r` only from `u` to `d` would incorrectly replace `d^3-s^3` by `d^3-u^3`.

## Limits and qualifications

At `s=0`, `B_lo=E_D=0`, and the exact quadratic coefficient remains

```text
J(u,d,0) = (8/15)*u^2*(15*d^4 + 10*d^2*u^2 - 8*d*u^3 + 8*u^4).
```

The ordinary nonzero-Fermi-surface Sommerfeld theorem does not apply directly to the strange factor at that endpoint. Nevertheless, the angular kernel behaves as `p_s^3` at small outgoing strange momentum. Therefore smoothing the strange lower endpoint first changes the integral at `T^4`, leaving the displayed `T^2` and `T^3` coefficients intact. The coefficient of `T^4` does require the endpoint treatment.

For `d=u+epsilon`, `s=u-epsilon`, the polynomial gives

```text
J = (2048/15)*u^5*epsilon + O(epsilon^2)
  = (1024/15)*u^5*(d-s) + higher flavor-splitting terms.
I0 = (128/45)*u^5*(d-s)^3 + higher flavor-splitting terms.
```

Consequently the leading joint degenerate/equal-flavor limit of the net rate is

```text
Gamma_K = 16*G_F^2*sin(theta_C)^2*cos(theta_C)^2/(5*pi^5)
          * u^5*(d-s)*[(d-s)^2 + 4*pi^2*T^2]
          + higher terms in the small ratios to the flavor Fermi scale.
```

This is the expected Eq. (23) normalization check. It is a separate uniform Fermi-surface reduction; the fixed-imbalance forward expansion by itself does not supply a controlled error estimate when `T` is comparable to any vanishing flavor splitting. For an actual finite-temperature EOS trajectory, evaluate `I0` and the coefficient at the local chemical potentials; a Taylor expansion along that trajectory would also include the temperature-induced shifts of those chemical potentials.

At exactly `u=d=s`, the net rate is identically zero, while the forward rate starts at a positive `T^3` term. Therefore the displayed fixed-imbalance cubic coefficient must not be evaluated at that fully degenerate point by continuity. Additional coincident flavor thresholds at `s=u<d` or `s<d=u` also require their own cubic analysis. `phase_space_t3` deliberately requires strict `s<u<d`, while allowing `s=0`.

## Wolfram and independent checks

The following actual Wolfram input checked the nonanalytic coefficient:

```wolfram
Clear[x,n,u,d,s,e,m];
w[e_,m_]:=32m^2(e-m)^2;
{Integrate[2 x^3 (x Coth[x/2]-2)/(4 Sinh[x/2]^2),{x,0,Infinity}],
 Factor[Integrate[w[u+r,r],{r,s,u}]+Integrate[w[u+r,u],{r,u,d}]],
 FullSimplify[2 Sum[n^2*Factorial[4]/n^5-2n*Factorial[3]/n^4,
                    {n,1,Infinity}]]}
```

The returned output was exactly

```text
{24*Zeta[3], 32*(d-s)*(d^2+d*s+s^2)*u^2/3, 24*Zeta[3]}.
```

Wolfram separately evaluated all six boundary integrals, returning the displayed closed polynomial, the branch difference

```text
J(q>0)-J(q<=0) = -8*(d-s-u)^4
                  *(9*d^2+4*d*s+2*s^2+4*d*u+4*s*u+2*u^2)/15,
```

and the limits `J(u,d,0)` and `2048*epsilon*u^5/15` above. The same polynomial integrations are reproduced independently with SymPy in `thermal_sympy.py`; its exact branch-difference assertion passes.

At `u=1`, neutral `a=0.146`, the helper returns `J=6.062655501599992`, the quadratic coefficient `9.972668570146649`, and the cubic coefficient `-7.488013135415503`. Independent finite-temperature quadrature gave `(I(T)-I0-C2*T^2)/T^3 = -8.7181160, -7.8978937, -7.6109634, -7.5290003` at `T=0.003,0.001,0.0003,0.0001`, approaching the predicted cubic limit with a residual linear in `T`.

At the neutral endpoint `u=1,d=2^(1/3),s=0`, the coefficients are `C2=45.26180717832831` and `C3=-51.28776120147602`. The same independent residual at `T=0.003,0.0015,0.00075,0.000375` is `-49.5015088912,-50.3946862826,-50.8412344294,-51.0644922832`. After also subtracting the cubic term, the residual divided by `T^4` is `595.4174368,595.3832792,595.3690294,595.3837822`, confirming a finite quartic endpoint correction. These numerical data are recorded in `results/low-temperature-residuals.csv` by the independent rate-validation script.
