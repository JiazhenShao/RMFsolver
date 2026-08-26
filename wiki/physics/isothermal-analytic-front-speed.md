---
summary: The fixed-temperature speed uses the local K fraction; along the representative isobar, neglecting the baryon-density derivative changes a' by at most 0.47%.
status: current
updated: 2026-08-25
tags: [physics, analytic, front-speed, isothermal]
---

# Isothermal analytic front speed

Replacing energy-flux conservation by $T(x)=\mathrm{const}$ removes the temperature trajectory from the [[analytic-front-speed|piecewise-constant speed estimate]].
With the cubic-plus-thermal-linear weak source and constant quark-side coefficients, the positive incoming speed is

$$
u(0^-)=\frac{a(0^+)}{\lambda_n}
\sqrt{\frac{D_K\gamma_K\left(a^2(0^+)+2\eta\right)}
{2[1-a(0^+)][1+\xi a(0^+)]}},
$$

where $\lambda_n\equiv n_B(0^-)/n_B(\infty)$, $-1<\xi<1$, and

$$
\eta=\frac{9\pi^2T^2}{\mu_q^2}.
$$

Throughout the analytical derivation, the composition is the **local** K fraction

$$
a(x)\equiv\frac{n_K(x)}{n_B(x)}.
$$

Pure neutron matter has $a(x<0)=1$ because $n_K=(n_d-n_s)/2=n_B$ there.
The sharp interface does not require $a(0^+)=a(0^-)$; only the K current is continuous, so conservation laws alone do not determine $a(0^+)$.
The public solvers therefore either accept it from the caller or select the separate static-isobar thermodynamic ceiling described below.
The factor $1-a(0^+)$ in the speed formula is the explicit use of $a(0^-)=1$ in that current jump.

`analytic_velocity_isothermal` obtains the supplied upstream PNM state directly at fixed $n_B(0^-)$ and reuses the accepted RMF solution for $\mu_B(0^-)$, $P(0^-)$, and $e(0^-)$.
It does not perform the former broad $\mu_B$ scan; branch safety now comes from scaled-residual rejection, two-seed agreement, and a final forward-density check as recorded in [[pnm-density-state-recovery]].

Thus the equivalent explicitly temperature-dependent numerator is $a^2(0^+)+18\pi^2T^2/\mu_q^2$.
The formula is exact in $I_2$ within the stated cubic-plus-linear weak-rate reduction; $\xi$ is the only conversion-profile input.
The static-isobar Taylor ceiling and its substituted maximum-speed expression are derived in [[isothermal-maximum-speed]].

## Fixed-isobar density drift

At fixed $(P,T)$, Gibbs--Duhem gives $d\mu_B/d\mu_K=-n_K/n_B=-a$.
For the representative profile with $T=10$ MeV, $n_B(0^-)=3n_0$, $B^{1/4}=189.2$ MeV, and $a(0^+)=0.146$, both $\mu_B$ and $n_B$ rise by about $0.24\%$ between $0^+$ and infinity.
The exact local-fraction derivative is $a'=n_K'/n_B-a n_B'/n_B$.
Using $a'\simeq n_K'/n_B$ changes the derivative by at most $0.47\%$ along this trajectory, which supports dropping the density-derivative correction in the body of the analytical treatment.

## Integrated K-ness equation

Under the piecewise-constant quark-side background,

$$
J_K=n_B(\infty)\left[u(\infty)a-D_Ka'\right],
\qquad
\frac{dJ_K}{dx}=-n_B(\infty)\gamma_K(a^3+\eta a),
$$

with $u(\infty)=\lambda_nu(0^-)$.  Therefore

$$
\frac{d}{dx}(D_Ka')=u(\infty)a'+\gamma_K(a^3+\eta a).
$$

Multiplication by $D_Ka'$ and integration from $0^+$ to $\infty$ give

$$
\frac12[D_Ka'(0^+)]^2=-u(\infty)I_1+I_2,
$$

where

$$
I_1=\int_{0^+}^{\infty}D_K(a')^2\,dx,
\qquad
I_2=D_K\gamma_K\int_0^{a(0^+)}(a^3+\eta a)\,da.
$$

Because $D_K$, $\gamma_K$, and $\eta$ are constant, the second integral is an endpoint expression:

$$
I_2=D_K\gamma_K\left[\frac{a^4(0^+)}{4}+\frac{\eta a^2(0^+)}{2}\right]
=\frac{D_K\gamma_Ka^2(0^+)[a^2(0^+)+2\eta]}{4}.
$$

For $g(a)\equiv-D_Ka'>0$, the interface condition fixes

$$
g[a(0^+)]=\lambda_nu(0^-)[1-a(0^+)],
$$

and $g(0)=0$ downstream.  The profile area is parameterized by

$$
I_1=\int_0^{a(0^+)}g(a)\,da
=\frac{1+\xi}{2}\lambda_nu(0^-)[1-a(0^+)]a(0^+).
$$

Substitution gives

$$
I_2=\frac12\lambda_n^2u^2(0^-)[1-a(0^+)][1+\xi a(0^+)],
$$

which yields the boxed speed above after selecting the positive root.

## ⚠ Local-fraction current-jump form

When $a=n_K/n_B$ is used on both sides of the interface, the upstream term in the fraction-gradient jump must carry the density ratio explicitly.
Baryon conservation gives

$$
n_B(0^-)u(0^-)=n_B(\infty)u(\infty)=j_B,
\qquad
u(\infty)=\lambda_nu(0^-).
$$

K-current continuity on the piecewise-constant downstream background therefore gives

$$
D_Ka'(0^+)
=u(\infty)[a(0^+)-a(0^-)]
=\lambda_nu(0^-)[a(0^+)-a(0^-)].
$$

The integrated equation can equivalently be written

$$
\frac12\left\{\lambda_nu(0^-)[a(0^+)-a(0^-)]\right\}^2
=\frac{1+\xi}{2}\lambda_nu(0^-)a(0^+)
\left\{\lambda_nu(0^-)[a(0^+)-a(0^-)]\right\}+I_2.
$$

Solving it before imposing pure neutron matter gives

$$
u^2(0^-)=\frac{2I_2}
{\lambda_n^2[a(0^-)-a(0^+)][a(0^-)+\xi a(0^+)]}.
$$

For $a(0^-)=1$, this is exactly the implemented isothermal formula.
If the factor $\lambda_n$ is omitted from the upstream local-fraction term, the denominator instead becomes $[a(0^-)-\lambda_na(0^+)][a(0^-)+\xi\lambda_na(0^+)]$, which is generally different and is not the implemented formula.

## Relation to the numerical isothermal BVP

`solve_front_isothermal` now propagates the physical K density and current,

$$
\frac{dn_K}{dx}=\frac{u n_K-J_K}{D_K},
\qquad
\frac{dJ_K}{dx}=-\Gamma_K(\mu_B,\mu_K,T,m_s),
$$

and reports the composition only as the derived local fraction

$$
a(x)=\frac{n_K(x)}{n_B(x)}.
$$

For each trial baryon current $j_B$, the fixed upstream state gives $u(0^-)=j_B/n_B(0^-)$ and $\Pi=P(0^-)+h(0^-)u^2(0^-)$.
The solver reconstructs $n_B(x)$, $u(x)$, $\mu_B(x)$, and $\mu_K(x)$ pointwise from fixed $T$, $j_B$, $\Pi$, and the current value of $n_K$.
It then recomputes both the exact nonleptonic source $\Gamma_K$ and $D_K[\mu_B(x),T]$ at every BVP node.

The upstream local fraction for a nuclear proton fraction $Y_p$ is

$$
a(0^-)=1-\frac{Y_p}{2}.
$$

K-current continuity therefore imposes

$$
J_K(0^+)=j_Ba(0^-),
$$

while the supplied $a(0^+)$ fixes $n_K(0^+)$ through the local quark EOS.
The downstream state satisfies $\mu_K(\infty)=0$; for finite $m_s$, both $n_K(\infty)$ and $J_K(\infty)=u(\infty)n_K(\infty)$ can be nonzero.

The semi-infinite layer is compactified with

$$
s=1-\exp\left[-\frac{\lambda_\infty x}{\kappa_{\rm factor}}\right],
\qquad
0\leq s\leq1-\epsilon_{\rm tail},
$$

where $\lambda_\infty$ is obtained by linearizing the same exact fixed-$T$ closure at equilibrium.
A shifted Robin condition matches the finite-$m_s$ tail to $[n_K(\infty),J_K(\infty)]$.

This numerical system is more exact than the analytical formula above.
The analytical derivation deliberately freezes the background coefficients and reduces the source to $\gamma_K(a^3+\eta a)$ so that $I_2$ is an endpoint polynomial; the numerical BVP makes neither approximation.

## Formal zero-temperature limit

Assume that $a(0^+)$, $\lambda_n$, $\mu_q$, $\gamma_K$, and $\xi$ approach finite constants with $0<a(0^+)<1$ as $T\to0^+$.  The full diffusion coefficient obeys

$$
D_K^{-1}=\frac{8N_f\alpha_s^2}{\pi}
\left(h_D\frac{T^{5/3}}{q_D^{2/3}}+\frac{\pi^3T^2}{12q_D}\right),
$$

so its leading behavior is

$$
D_K\sim\frac{\pi q_D^{2/3}}{8N_f\alpha_s^2h_D}\,T^{-5/3}.
$$

Meanwhile $\eta\propto T^2\to0$, but the suprathermal $a^3$ weak source remains nonzero.  Therefore

$$
u(0^-)\sim
\frac{a^2(0^+)}{4\alpha_s\lambda_n}
\sqrt{\frac{\pi q_D^{2/3}\gamma_K}
{N_fh_D[1-a(0^+)][1+\xi a(0^+)]}}
\,T^{-5/6}.
$$

Thus the fixed-composition mathematical limit diverges:

$$
\lim_{T\to0^+}u(0^-)=\infty.
$$

The exact point $T=0$ is not an admissible state of `solve_front_isothermal`, whose microphysics guard requires $T>0$; this result is only the one-sided asymptotic limit.  It is also distinct from the energy-conserving problem with only $T(0^+)=0$, because the isothermal construction sends the entire quark-side profile to zero temperature.

This is not a physical infinite front speed.  The same scaling gives a conversion length $L_{\rm conv}\propto\sqrt{D_K}\propto T^{-5/6}$, while the transport mean free path grows as $D_K\propto T^{-5/3}$.  Hence the Knudsen ratio grows as $T^{-5/6}$ and the local diffusion/LTE description fails before zero temperature; the assumed nonrelativistic slow-front regime also fails once $u(0^-)$ is no longer small.

The joint limit is path dependent if $a(0^+)$ also tends to zero.  For $a(0^+)\propto T^p$ with $0<p<1$, the cubic source dominates and

$$
u(0^-)\propto T^{2p-5/6}.
$$

It diverges for $p<5/12$, approaches a finite nonzero value for $p=5/12$, and vanishes for $p>5/12$.  Thus the isothermal equations do not define a unique zero-temperature velocity unless the interface closure specifies the limiting behavior of $a(0^+)$.

## Stable neutron-matter region and zero-speed boundary

For $T>0$, $D_K>0$, $\gamma_K>0$, $\eta>0$, $0\leq a(0^+)<1$, and $-1<\xi<1$, the analytical speed satisfies

$$
u(0^-)=0\quad\Longleftrightarrow\quad a(0^+)=0.
$$

Thus the reaction--diffusion formula alone does not locate a finite-density phase-stability boundary.  The [[interface-closure|energy-conserving]] condition $\Delta h=0$ must not be reused after energy conservation has been replaced by isothermality.

For a static isothermal interface, bulk coexistence instead requires a common temperature, baryon chemical potential, and pressure, with equilibrated quark matter:

$$
T_{\rm NM}=T_{\rm QM}=T,
\qquad
\mu_{B,\rm NM}=\mu_{B,\rm QM},
\qquad
P_{\rm NM}=P_{\rm QM},
\qquad
\mu_K=0.
$$

For a contour parameterized by the upstream density, first obtain $\mu_B(0^-)$ from the nuclear EOS and define

$$
\Delta P_{\rm iso}[T,n_B(0^-)]
=P_{\rm NM}[\mu_B(0^-),T]
-P_{\rm QM}[\mu_B(0^-),0,T].
$$

At common $T$ and $\mu_B$, the phase with larger pressure is thermodynamically favored.  Therefore

$$
\begin{array}{lll}
\Delta P_{\rm iso}>0 &: & \text{neutron matter stable; no forward conversion front},\\
\Delta P_{\rm iso}=0 &: & \text{isothermal coexistence and }u(0^-)=0,\\
\Delta P_{\rm iso}<0 &: & \text{quark matter favored; neutron matter can be metastable}.
\end{array}
$$

The stable region should be masked or hatched rather than filled by evaluating the positive-speed formula.  Setting its forward speed to zero is a plotting convention; inside that region the modeled neutron-to-quark front does not exist, and the reverse process is outside the present calculation.

The implemented density-parameterized classifier uses the equivalent common-$(P,T)$ construction.
Let $\mu_{B,\rm QM}^{(P)}$ solve

$$
P_{\rm QM}[\mu_{B,\rm QM}^{(P)},0,T]=P(0^-),
$$

and define

$$
\Delta\mu_B\equiv\mu_{B,\rm QM}^{(P)}-\mu_B(0^-).
$$

At common $P$ and $T$, the phase with lower baryon chemical potential is favored, hence

$$
\begin{array}{lll}
\Delta\mu_B>0 &: & \text{neutron matter stable; no forward front},\\
\Delta\mu_B=0 &: & \text{isothermal coexistence},\\
\Delta\mu_B<0 &: & \text{quark matter favored; solve the moving branch}.
\end{array}
$$

This has the same stability sign as $\Delta P_{\rm iso}$ above but avoids evaluating both pressures at a common trial chemical potential when the contour input is $n_B(0^-)$.

A direct diagnostic scan of the live QMC-RMF3 and massless bag-model EOS gives the following neutron-side coexistence densities:

| $T$ (MeV) | $n_{B,\rm coexist}(T)/n_0$ for $B^{1/4}=180$ MeV | $n_{B,\rm coexist}(T)/n_0$ for $B^{1/4}=189.1566$ MeV |
|---:|---:|---:|
| 0 | 2.3300 | 3.0000 |
| 20 | 2.1028 | 2.8335 |
| 40 | 1.3768 | 2.2923 |
| 60 | 0.5246 | 1.3247 |

For these branches, neutron matter is stable on the lower-density side of the tabulated curve.  The second calibration is the one chosen so cold coexistence occurs at $3n_0$.

## Analytical API

`analytic_velocity_isothermal(T_0minus, nB_0minus, B_one_forth, a_0plus=np.nan, *, xi=0, ...)` implements the local-fraction formula as a finite-flux scalar eigenvalue.
An explicit finite `a_0plus` evaluates that prescribed interface composition.
Omitting it activates the PNM-only automatic ceiling defined by

$$
P(0^+)=P(0^-),\qquad
a(0^+)=\frac{n_K(0^+)}{n_B(0^+)},\qquad
\mu_B(0^-)=\mu_B(0^+)+a(0^+)\mu_K(0^+).
$$

An interior root reports `a_0plus_source="maximum"` and `a_0plus_max_status="interior"`.
If the ceiling reaches the exact endpoint $a(0^+)=1$, the function returns `status="composition_ceiling_saturated"` and no speed; the singular endpoint is never evaluated as a moving front.
For every trial $u(0^-)$ it constructs $j_B=n_B(0^-)u(0^-)$ and $\Pi=P(0^-)+h(0^-)u^2(0^-)$, solves the equilibrated $\mu_K(\infty)=0$ endpoint, and solves the $0^+$ state from $\Pi$, $j_B$, and $n_K(0^+)/n_B(0^+)=a(0^+)$.
It then finds the root of $u_{\rm formula}^2(0^-)-u^2(0^-)$.
The piecewise-constant density ratio uses $n_B(\infty)$, so $\lambda_n=n_B(0^-)/n_B(\infty)$, while the analytical transport coefficients are deliberately frozen at $0^+$.
The exact hydro states can have $n_B(0^+)\neq n_B(\infty)$; retaining both exact states while using this mixed constant-background reduction is the explicit analytical approximation, not an assertion that the two densities are identical.

The function returns `u_0minus=0` with `front_exists=False` in stable neutron matter and at coexistence.
At exactly $T=0$, it still performs that thermodynamic classification; a quark-favored point with $a(0^+)>0$ returns `status="zero_temperature_transport_invalid"` and no finite velocity.
At positive $T$, if the formal eigenvalue does not occur within the slow-front domain $u(0^-)<1$, it returns `status="slow_front_approximation_invalid"` instead of following a disconnected high-velocity EOS branch.
Even below that root-search ceiling, a resolved result is masked with `status="momentum_flux_ratio_above_tolerance"` when $(\Pi-P(0^-))/P(0^-)$ exceeds the default $10^{-3}$ analytical-validity threshold.
The present derivation and implementation require `NM_type="PNM"` and `ms=0`.
A nonzero strange-quark mass gives $n_K(\infty)\neq0$ and requires rewriting the weak source in terms of departure from equilibrium; other upstream compositions need a corresponding replacement for $a(0^-)=1$.

## ⚠ Why the coefficients are constant

Isothermality by itself does not make $D_K$ independent of position because [[strangeness-reaction-diffusion|the transport coefficient]] also depends on $q_D\propto\mu_q$.
The coefficients are constant only in the analytical reduction because it also freezes $\mu_q$.
The numerical `solve_front_isothermal` instead updates $D_K$ with the local $\mu_B(x)$ and evaluates the exact $\mu_K$-dependent rate at every node.
Consequently, its source integral is not the polynomial endpoint expression $I_2$ above.

## Symbolic check

Wolfram Language evaluated

```text
Integrate[dk*gammak*(a^3 + eta*a), {a, 0, a0}]
```

as `a0^2*dk*gammak*(a0^2 + 2*eta)/4`.
It also simplified the final balance residual to zero and returned

```text
u2 = a0^2*dk*gammak*(a0^2 + 2*eta) /
     (2*lambda^2*(1 - a0)*(1 + xi*a0))
```

for the piecewise-constant limit.
For the zero-temperature check, Wolfram also returned

```text
lim_(T->0+) T^(5/3)*u2 =
Pi*qd^(2/3)*gamma*a0^4 /
(16*nf*alpha^2*hd*lambda^2*(1-a0)*(1+xi*a0))
```

For the finite-temperature zero-speed question, Wolfram returned `a0 == 0` as the only root under the physical sign and domain assumptions.
For the local-fraction current-jump check, the Wolfram Language input was

```text
Solve[
  1/2 lambdaN^2 v (aPlus-aMinus)^2 ==
    (1+xi)/2 lambdaN^2 v aPlus (aPlus-aMinus) + I2,
  v
]
```

and returned `v -> 2 I2/[(aMinus-aPlus) lambdaN^2 (aMinus+xi aPlus)]`.

## Velocity convention and the static-isobar bound

The variable the solvers call $u$ is the **proper** velocity $\gamma v$, not the 3-velocity.
The module's own `_relativistic_gamma_from_u` (15 call sites) has always returned $\gamma=\sqrt{1+u^2}$, which is the same statement.

This makes the isothermal junction conditions exact as written.
With $x\equiv j_B/n_B$, inverting $j_B=n_B\gamma v$ gives $v=x/\sqrt{1+x^2}$, hence $\gamma v=x$ and

$$
h\gamma^2v^2=hx^2.
$$

So $j_B=n_Bu$ and $\Pi=P+hu^2$ are **already** the exact relativistic baryon and momentum fluxes, not a $\gamma\to1$ reduction of them.
Verified symbolically and numerically: an opt-in "relativistic" branch built on 2026-08-19 shifted results by $0$ to $2.4\times10^{-15}$, i.e. it was an algebraic identity, and it was reverted.

⚠ **The trap.** Reading $u$ as a 3-velocity makes $\gamma=1/\sqrt{1-u^2}$, which is wrong here and inflates $\gamma$; it also suggests a relativistic correction that does not exist.
That misreading produced a whole implementation plan before it was caught.

**What *is* approximate.** `_solve_a_0plus_max` evaluates the interface in the static limit, $j_B=0$ and $\Pi=P(0^-)$, dropping the flux term.
Its relative size is $(\Pi-P)/P=hu^2/P$, which scales as $1/[1-a(0^+)]$ because $u(0^-)\propto[1-a(0^+)]^{-1/2}$:

| $T$ (MeV) | $n_B(0^-)/n_0$ | $h/P$ | $1-a(0^+)$ at 10% | $\gamma-1$ there |
|---:|---:|---:|---:|---:|
| 1 | 3.5 | 5.74 | $10^{-9}$ | 0.019 |
| 40 | 3.0 | 6.80 | $10^{-11}$ | 0.0075 |
| 80 | 2.0 | 8.74 | $3\times10^{-12}$ | 0.039 |

Note $h/P\approx6$–$9$: the enthalpy density is several times the pressure, not below it.
Realized compositions span $a_{\max}(0^+)\in(0.01,0.99)$, where the correction is under $10^{-6}$ — below `tol_bvp`, the ray solver's $\rho$ tolerance, and the momentum-flux residual gate alike.
`MOMENTUM_FLUX_RATIO_TOLERANCE = 1e-3` makes that failure loud rather than silent; it does not bind anywhere in the present domain.

**Open.** `slow_front_consistent` tests $u(0^-)<1$, but for a proper velocity that is $v<0.707$, not $v<1$ — the label and the bound disagree.

The boundary-fitted analytical and exact numerical contour workflow that uses this automatic ceiling is recorded in [[isothermal-contour-cluster]].

## Sources

- `RMFsolver/phase_velocity.py`, `analytic_velocity_isothermal`, `solve_front_isothermal`, `_exact_kaon_transport_rate`, and the fixed-$T$ local EOS closure helpers.
- `Latex_writting/Hydrodynamical combustion/Notes.tex`, “Isothermal Notes (Mar 16, 2026).”
- [[analytic-front-speed]] for the corresponding non-isothermal construction.
