# Exact cold massless weak rate

This is an independent evaluation of the massless four-quark phase-space integral for the positive `d+u -> s+u` direction, with no QCD correction or additional antiquark channels.
The physical ordering is `0 <= mu_s < mu_u < mu_d`; the formulas extend continuously to equality.
The reverse reaction is Pauli blocked at zero temperature because its occupied incoming energies sum to at most `mu_u+mu_s`, below the minimum outgoing sum `mu_u+mu_d`.
Therefore the forward rate is also the net rate.

The following ASCII notation is for reproducible algebra only: `u=mu_u`, `d=mu_d`, `s=mu_s`.
In physical prose, retain the chemical potentials rather than introducing these as new thermodynamic variables.
The normalization established in the companion derivation is

```text
Gamma0 = [9*G_F^2*sin(theta_C)^2*cos(theta_C)^2/(8*pi^5)] * I0
K(E,m) = 32*E^2*m^3/3 - 16*E*m^4 + 32*m^5/5
I0 = integral_(u+s)^(u+d) dE
     integral_max(0,E-d)^u dp1 integral_s^(E-u) dps K(E,min(p1,ps)).
```

Here `p1` and `ps` are the incoming up and outgoing strange momenta.
The other outgoing up momentum satisfies `p2 >= u >= p1`; consequently the minimum of all four momenta is `min(p1,ps)`.

## Integration order and the only branch boundary

Change the integration order to integrate the total energy first:

```text
max(0,u+s-d) <= p1 <= u,
s <= ps <= p1+d-u,
u+ps <= E <= p1+d.
```

Since `d>u`, the upper strange-momentum bound `p1+d-u` is always above `p1`.
Also `max(0,u+s-d) <= s`.
Thus precisely three polynomial regions suffice:

```text
1. max(0,u+s-d) <= p1 <= s, s <= ps <= p1+d-u, m=p1;
2. s <= p1 <= u, s <= ps <= p1, m=ps;
3. s <= p1 <= u, p1 <= ps <= p1+d-u, m=p1.
```

All three use the same energy bounds `u+ps <= E <= p1+d`.
Only the first region changes polynomial when `d-s` crosses `u`.
Apparent additional boundaries such as `d=2*u` or `d+s=2*u`, encountered with the original energy-first ordering, do not produce new final branches.
This argument covers arbitrary `d>u`, not just the neutral range.

For example, the first region after the energy and strange-momentum integrations has the polynomial integrand

```text
8*p1^3*(d-s-u+p1)^2/45
  * [15*d^2 + 10*d*s + 10*d*u + 5*s^2 + 10*s*u
     - 5*s*p1 + 5*u^2 - 5*u*p1 + 3*p1^2].
```

Integrating this from `p1=0` instead of the correct positive lower bound overcounts the smaller-imbalance branch.
The overcount is the compact sixth-power correction below.

## Explicit exact result

The common large-imbalance polynomial can be written compactly as follows; `P` is only a displayed polynomial, not a new physical function.

```text
315*P = u^4*(210*d^4 - 112*d^3*u - 84*d^2*u^2 - 32*d*u^3 + 65*u^4)
      + s^4*(168*d^3*s - 280*d^3*u - 168*d^2*s^2 + 504*d^2*s*u
             - 420*d^2*u^2 + 80*d*s^3 - 336*d*s^2*u + 504*d*s*u^2
             - 280*d*u^3 - 18*s^4 + 64*s^3*u - 196*s^2*u^2
             + 168*s*u^3 + 210*u^4).

I0 = P - (u+s-d)^6*[15*d^2 + 10*d*(s+u) + 3*(s+u)^2]/315,
     when 0 < d-s <= u;

I0 = P,
     when d-s >= u.
```

Equivalently, use `max(u+s-d,0)^6` in the correction and a single formula.
Every term has energy degree eight.
At `d=s+u`, the two branches agree through derivatives of order five.
For derivatives with respect to `d` at fixed `u,s`, the large-branch polynomial minus the continued small-branch polynomial has sixth derivative `64*(s+u)^2` at the boundary.
Thus the piecewise phase-space rate is `C^5` but is not generally `C^6` there.

The individual terms `P` and the correction nearly cancel near equal flavor potentials.
Do not use this compact representation for ordinary double-precision evaluation very close to equilibrium.
`cold_sympy.py::cold_phase_space` instead expands the small-imbalance branch in `d-u` and `u-s`, factors out `u^5*(d-s)^3`, and evaluates the residual dimensionless polynomial.
Its leading term is

```text
I0 = (128/45)*mu_u^5*(mu_d-mu_s)^3 + higher total orders
     in (mu_d-mu_u) and (mu_u-mu_s).

Gamma0 = [16*G_F^2*sin(theta_C)^2*cos(theta_C)^2/(5*pi^5)]
         *mu_u^5*(mu_d-mu_s)^3 + higher orders.
```

This is a joint near-equal-flavor limit within the stated ordering, not a general expansion in imbalance at arbitrary fixed `mu_u`.

## Neutral specialization

On the massless neutral cold EOS,

```text
mu_d^3 + mu_s^3 = 2*mu_u^3,
mu_d/mu_u = (1+a)^(1/3),
mu_s/mu_u = (1-a)^(1/3),
0 <= a <= 1.
```

The branch crossing is uniquely determined by

```text
(1+a)^(1/3) - (1-a)^(1/3) = 1.
a = 0.9831494087020976555863106458883163...
mu_s/mu_u = 0.2563726633091643120118672630890943...
```

Use the smaller-imbalance branch below this value and the larger-imbalance branch above it.
At `a=0` the net rate is zero; the ratio to the leading cold approximation is defined by its limit, one.

| `a` | Exact cold rate / cold Eq. (23) |
| --- | --- |
| 0.001 | 0.999570090274 |
| 0.146 | 0.932490510812 |
| 0.5 | 0.724990158532 |
| 0.9 | 0.359423647541 |
| 0.99 | 0.189682513772 |
| 1 | 0.109658303010 |

These reproduce the earlier independent phase-space quadrature benchmarks to its quoted numerical accuracy.
The one-sided neutral near-equilibrium ratio has the especially simple expansion

```text
Gamma0 / Gamma_Eq23_cold = 1 - 55*a/128 - 2*a^2/9 - a^3/288 + O(a^4),
a -> 0+.
```

The exact zero-strangeness endpoint is

```text
I0(mu_s=0) = mu_u^4/315
  * [210*mu_d^4 - 112*mu_d^3*mu_u - 84*mu_d^2*mu_u^2
     - 32*mu_d*mu_u^3 + 65*mu_u^4].

Gamma0 / Gamma_Eq23_cold =
  [210*mu_d^4 - 112*mu_d^3*mu_u - 84*mu_d^2*mu_u^2
   - 32*mu_d*mu_u^3 + 65*mu_u^4] / (896*mu_d^3*mu_u).
```

At the neutral endpoint `mu_d=2^(1/3)*mu_u`, the ratio is `0.1096583030099788`.

## Reproduction and verification

Run from the repository root:

```bash
./bin/python3 ref-list/massless-weak-rate/cold_sympy.py
```

The script carries out exact SymPy polynomial integrations and verifies the branch correction, matching derivatives, zero-strangeness endpoint, rational points on both branches (including `d>2*u`), and stable near-equilibrium evaluation.
Import `cold_phase_space(mu_u,mu_d,mu_s)` for a floating-point phase-space integral without the weak prefactor.
This callable needs only the standard library; SymPy is imported only for derivation and verification.
The companion callable `cold_phase_space_gradient(mu_u,mu_d,mu_s)` returns the ordered tuple `(dI0/dmu_u,dI0/dmu_d,dI0/dmu_s)` for EOS chain-rule calculations.
It lazily compiles exact SymPy derivatives once and uses an expansion in `mu_d-mu_u` and `mu_u-mu_s` on the small-imbalance branch.
Unlike the rate evaluator, the gradient evaluator requires SymPy at its first call.

An independent Wolfram Language evaluation on 2026-10-07 used the following full input:

```wolfram
Clear[x,y,en,m,u,d,s];
k=32 en^2 m^3/3-16 en m^4+32 m^5/5;
j=Integrate[k,{en,u+y,x+d}];
low=Integrate[j/.m->x,{y,s,x+d-u}];
common=Integrate[Integrate[j/.m->y,{y,s,x}],{x,s,u}]
       +Integrate[Integrate[j/.m->x,{y,x,x+d-u}],{x,s,u}];
small=Factor[Integrate[low,{x,u+s-d,s}]+common];
large=Factor[Integrate[low,{x,0,s}]+common];
p=(u^4(210d^4-112d^3 u-84d^2 u^2-32d u^3+65u^4)
   +s^4(168d^3 s-280d^3 u-168d^2 s^2+504d^2 s u-420d^2 u^2
        +80d s^3-336d s^2 u+504d s u^2-280d u^3-18s^4
        +64s^3 u-196s^2 u^2+168s u^3+210u^4))/315;
correction=(u+s-d)^6(15d^2+10d(s+u)+3(s+u)^2)/315;
{Factor[large-p],Factor[small-(p-correction)],
 Table[Factor[D[large-small,{d,n}]/.d->u+s],{n,0,6}],
 Factor[45 (large/.s->0)/(128u^5 d^3)]}
```

Wolfram returned:

```text
{0, 0, {0,0,0,0,0,0,64*(s+u)^2},
 (210*d^4-112*d^3*u-84*d^2*u^2-32*d*u^3+65*u^4)/(896*d^3*u)}
```

For the neutral series, with `p` and `correction` as above, the additional Wolfram input was

```wolfram
Series[45(p-correction)/(128u^5(d-s)^3)
       /.{u->1,d->(1+a)^(1/3),s->(1-a)^(1/3)},{a,0,3}]
```

It returned `SeriesData[a,0,{1,-55/128,-2/9,-1/288},0,4,1]`, agreeing with SymPy.
In an additional floating-point audit, 200 random ordered triples and 13 near-equilibrium triples were compared against the exact polynomials evaluated at 60 decimal digits.
The largest relative difference was `3.78e-15`.

The ordered integration regions justify the branch geometry; computer algebra verifies the polynomial integrations and algebraic equivalence, not the physical normalization independently.
The exact cold result alone does not determine the finite-temperature coefficient: independently varied incoming and outgoing up Fermi surfaces must be retained before constructing a Sommerfeld expansion.
