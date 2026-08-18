---
summary: The energy-conserving and isothermal fronts are reduced reaction-diffusion BVPs in physical nK and JK with pointwise EOS reconstruction.
status: current
updated: 2026-08-18
tags: [method, bvp, numerical, front]
---

# Reduced steady-front boundary-value problem

The ideal energy-conserving and isothermal calculations propagate the physical K density/current pair.
Baryon density, flow, and chemical potentials are reconstructed algebraically from the EOS and conserved fluxes; temperature is reconstructed in the energy-conserving branch and fixed in the isothermal branch.
This distinction resolves the recurring but incorrect “five first-order fields need five interface data” counting.

## Differential fields

For $x>0$,

$$\frac{dn_K}{dx}=\frac{un_K-J_K}{D_K},\qquad \frac{dJ_K}{dx}=-\Gamma_K.$$

The global baryon flux $j_B$ is the velocity eigenvalue in disguise:

$$
u(0^-)=\frac{j_B}{n_B(0^-)}.
$$

For a trial $j_B$, the known upstream state always fixes the momentum flux

$$
\Pi=h(0^-)u^2(0^-)+P(0^-).
$$

The energy-conserving branch additionally fixes

$$
E=h(0^-)u(0^-)\sqrt{1+u^2(0^-)}.
$$

At every BVP node, that branch reconstructs the quark state by solving the EOS closure together with

$$
j_B=n_Bu,
\qquad
\Pi=hu^2+P,
\qquad
E=hu\sqrt{1+u^2}.
$$

The isothermal branch replaces the energy equation with

$$
T(x)=T(0^-),
$$

and reconstructs the local state from $n_K$, $T(0^-)$, $j_B$, and $\Pi$.
Both branches then evaluate the local $D_K$ and exact $\Gamma_K(\mu_B,\mu_K,T,m_s)$ for the two ODEs.
In particular, fixed temperature does not make $D_K$ constant because $D_K$ also depends on $\mu_B$ through the quark chemical potential.

See [[diffusion-limited-front]], [[quark-matter-eos]], [[strangeness-reaction-diffusion]], and [[isothermal-analytic-front-speed]].

## Boundary and eigenvalue conditions

- The upstream nuclear proton fraction fixes $a(0^-)=1-Y_p(0^-)/2$ and $n_K(0^-)=a(0^-)n_B(0^-)$.
- K-current continuity fixes $J_K(0^+)=j_Ba(0^-)$ in the $J_K=un_K-D_Kn'_K$ convention.
- The ideal formulation supplies $a(0^+)$ (or $n_K(0^+)$) as a model input.
- The downstream endpoint is flavor equilibrated: $\mu_K(\infty)=0$.
- For $m_s=0$, this also gives $n_K(\infty)=J_K(\infty)=0$; at finite $m_s$, the equilibrated density and advective current can be nonzero.
- The energy-conserving branch selects the baryon current/velocity and downstream temperature so the spatial tail and downstream energy flux both match; the isothermal branch selects only the baryon current/velocity.

The active `Main.tex` presents the speed as the single physical eigenvalue; the more explicit energy-conserving algorithm uses $j_B$ and $T(\infty)$ as two scalar BVP parameters.
These are compatible descriptions because $T(\infty)$ is a thermodynamic endpoint variable needed to enforce energy flux, not a second observable speed.
The isothermal BVP has only the scalar parameter $j_B$ because the common temperature is supplied.

## Compactification and continuation

The semi-infinite domain is mapped with

$$
x=-L_0\ln(1-s),
\qquad
0\le s\le1-\epsilon_{\rm tail}.
$$

For `solve_front_isothermal`, $L_0=\kappa_{\rm factor}/\lambda_\infty$, where $\lambda_\infty$ comes from the downstream linearization of the same exact fixed-$T$ closure used inside the BVP.
The right boundary is a shifted Robin condition about $[n_K(\infty),J_K(\infty)]$, so it remains valid at finite $m_s$.

Solutions must be stable under changes of $L_0$ and $\epsilon_{\rm tail}$.
Some energy-conserving branches use interface-fraction continuation when a direct solve fails; the physical-field isothermal solver currently attempts the requested interface fraction directly.
Negative temperatures, nonpositive densities, failed EOS closures and nonphysical transport coefficients are rejected.

This is the manuscript-level method. Current implementation-specific continuation, low-temperature and tail behavior live under [[phase-velocity-overview]], [[unmax-low-temperature]], [[unmax-degeneracy]] and [[thermal-conducting]].

## Local-fraction convention

The active convention is exclusively

$$
a(x)=\frac{n_K(x)}{n_B(x)}.
$$

`solve_front_isothermal` now follows it in its inputs and profile payload while keeping $n_K$ and $J_K$ as the fundamental BVP fields.
Historical notes that normalize disequilibrium by a fixed downstream baryon density describe an older numerical convention and must not be mixed with the current API.

## Sources

- `Latex_writting/Hydrodynamical combustion/Main.tex`, first active document: “Hydrodynamic equations,” “Boundary conditions,” and “Numerical method.”
- `Latex_writting/Hydrodynamical combustion/Notes.tex`: “Numerically solving the Energy conserving set of equations” and the cleaner reduced-BVP block. Older entropy-flux formulations are historical and should not be merged into this method.
