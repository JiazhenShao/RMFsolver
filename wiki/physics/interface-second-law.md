---
summary: The second law across the flavor-conserving interface is the affinity mu_n >= mu_u + 2 mu_d; it caps a(0+) at 0.0064 and the T drop at 3.14 MeV (3n0, 10 MeV), and the code's isothermal ceiling violates it by 33 MeV/baryon.
status: open-question
updated: 2026-10-20
tags: [physics, closure, entropy, interface, bound]
---

# Second law across the deconfinement interface

Question (2026-10-20): the entanglement entropy of quarks bound in a neutron is "released" at deconfinement. Where does it go, and does that bound the temperature drop across $x=0$?
Not yet signed off by the user. The isothermal consequence below challenges the existing ceiling in [[isothermal-maximum-speed]].

## Where the entropy goes

No fixed amount of entanglement entropy is released.
Each confined quark's color state is maximally mixed ($\ln3$), but the deconfined medium is also a global color singlet, so each quark's color stays maximally mixed there too.
Cold neutron matter and cold quark matter are both pure ground states with zero thermodynamic entropy.
What deconfinement releases is free energy, the affinity of $n\to u+2d$.
Thermalization (Kaufman 2016; ETH) turns it into thermal entropy at the conserved energy.
At $3n_0$ and $T(0^-)=10$ MeV, the whole front (interface plus weak layer) produces only $s/n_B(\infty)-s/n_B(0^-)=0.5123-0.2656=0.247$ per baryon, which is below $\ln3=1.099$.
The Kharzeev–Levin "entanglement becomes Boltzmann entropy" picture (DIS, $S=\ln xG$) concerns a hard probe with abundant energy; it does not carry over to a slow, energy-limited front.

## The inequality

The first-order entropy current gives, to leading order in $u$, $T\,J_s=E-\sum_a\mu_aJ_a$, using total fluxes, so the result is frame-independent and includes any heat flux.
The interface conserves the baryon flux $j$, the $K$ flux (neutrons carry $K=1$ per baryon, so $J_K=j$) and the energy flux $E$. The second law $J_s(0^+)\ge J_s(0^-)$ then reads

$$
\frac{h/n_B-\mu_B(0^+)-\mu_K(0^+)}{T(0^+)}\;\ge\;\frac{h/n_B-\mu_n(0^-)}{T(0^-)}=\frac{s}{n_B}(0^-),
$$

with $\mu_B+\mu_K=\mu_u+2\mu_d$.
The left side equals $s/n_B(0^+)-\mu_K(1-a)/T(0^+)$. The second term is the entropy carried by the strangeness diffusion current $j(1-a)$.
Dropping it, i.e. requiring only $s(0^+)\ge s(0^-)$, gives a much weaker bound.

⚠ If $T(0^+)=T(0^-)$, the condition reduces to the interface affinity $\mu_n(0^-)\ge\mu_u(0^+)+2\mu_d(0^+)$. It is **not** the bulk Gibbs balance $\mu_n=\mu_B+a\mu_K$.

## Energy-conserving front: bound on the temperature drop

The state at $0^+$ lies on the conserved $(P,h/n_B)$ trajectory. In the quadratic EOS (Wolfram-verified), with $x=a/A$, the left side is $\sqrt2\pi(A-x)/\sqrt{1-x^2}$ and $s/n_B(\infty)=\sqrt2\pi A$, so

$$
1-\frac{a(0^+)}{A^2}\;\ge\;\frac{s/n_B(0^-)}{s/n_B(\infty)}\sqrt{1-\frac{a(0^+)^2}{A^2}} .
$$

Exact EOS, $3n_0$, $T(0^-)=10$ MeV, $A=0.1152$:

| bound | $a(0^+)$ max | $T(0^+)$ min | max drop $T(0^-)-T(0^+)$ |
|---|---|---|---|
| affinity (correct) | 0.006407 (quadratic: 0.006406) | 6.857 MeV | 3.143 MeV |
| thermal $s$ only (naive) | 0.0985 | 3.56 MeV | 6.44 MeV |

Energy conservation alone gives $T(0^+)\le T(\infty)=6.868$ MeV, so the interface drop is pinned to $3.132\le\Delta T\le3.143$ MeV.
The interface must be nearly strange-equilibrated ($a(0^+)\lesssim A^2$), so the formal maximization of $a(0^+)$ toward $A$ in [[analytic-front-speed]] is excluded long before the [[lte-composition-bound]] applies.

## Isothermal model: the existing ceiling destroys entropy

`_solve_a_0plus_max` imposes $\mu_n(0^-)=\mu_B(0^+)+a\,\mu_K(0^+)$. At that root the interface entropy production is exactly $-j(1-a)\mu_K/T<0$.
At $3n_0$ and $T=10$ MeV the code returns $a=0.1461$ with affinity $-33.18$ MeV per baryon. The affinity ceiling is $a=0.01075$.
Quadratic forms, with $\delta=\mu_n-\mu_B(0^+)|_{a=0}=2.834$ MeV: the affinity ceiling is $a\simeq9\delta/(2\mu_B)=0.0107$ and the code's ceiling is $3\sqrt{\delta/\mu_B}=0.146$, both matching the solver.
Because the speed rises with $a(0^+)$, isothermal maximum speeds built on the code ceiling are overestimates. That ceiling has not been changed in code (solver edits are restricted).

## Checks

The Euler $s/n_B$ matches `SQMsolver.entropyQM` along the trajectory to $2.3\times10^{-6}$. The quadratic closed forms match the exact numbers above. The isothermal affinity at the code root equals $-(1-a)\mu_K$ to $10^{-15}$.
Files: `local_analysis_results/26-10-20/deconfinement-quench/second_law_bound.py` and `.json`. See also [[deconfinement-quench-temperature]] and [[interface-closure]].

## Caveats

Slow-front $O(u^2)$ terms are dropped. The result assumes LTE at $0^\pm$ and massless quarks with no electrons.
The argument assumes $K$ is conserved through the layer, which is the model's own K-current matching. Weak reactions inside a microscopic layer are negligible.
