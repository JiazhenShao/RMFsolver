---
summary: The ideal energy-conserving front is a two-field reaction-diffusion BVP with global flux eigenvalues and pointwise EOS reconstruction, not a five-field initial-value problem.
status: current
updated: 2026-08-11
tags: [method, bvp, numerical, front]
---

# Reduced steady-front boundary-value problem

The ideal energy-conserving calculation propagates only the K density/current pair. Baryon density, flow, temperature and chemical potentials are reconstructed algebraically from the EOS and three conserved fluxes. This distinction resolves the recurring but incorrect “five first-order fields need five interface data” counting.

## Differential fields

For $x>0$,

$$\frac{dn_K}{dx}=\frac{un_K-J_K}{D_K},\qquad \frac{dJ_K}{dx}=-\Gamma_K.$$

The global baryon flux $j_B$ is the velocity eigenvalue in disguise:

$$u_N=\frac{j_B}{n_B^N}.$$

For a trial $j_B$, the known upstream state fixes

$$\Pi=h_Nu_N^2+P_N,\qquad E=h_Nu_N\sqrt{1+u_N^2}.$$

At every BVP node, the solver reconstructs the quark state by solving the EOS closure together with

$$j_B=n_Bu,\qquad \Pi=hu^2+P,\qquad E=hu\sqrt{1+u^2}.$$

It then evaluates $D_K$ and $\Gamma_K$ for the two ODEs. See [[diffusion-limited-front]], [[quark-matter-eos]] and [[strangeness-reaction-diffusion]].

## Boundary and eigenvalue conditions

- Upstream neutron matter fixes $n_K(0^-)=n_B^N$ and $n'_K(0^-)=0$.
- K-current continuity fixes $J_K(0^+)=u_Nn_B^N$ in the $J_K=un_K-D_Kn'_K$ convention.
- The ideal formulation supplies $a(0^+)$ (or $n_K(0^+)$) as a model input.
- The downstream endpoint is flavor equilibrated: $\mu_K(\infty)=n_K(\infty)=0$.
- The baryon current/velocity and downstream temperature are selected so the spatial tail and downstream energy flux both match.

The active `Main.tex` presents the speed as the single physical eigenvalue; the more explicit `Notes.tex` algorithm uses $(j_B,T_Q)$ as two scalar BVP parameters. These are compatible descriptions because $T_Q$ is a thermodynamic endpoint variable needed to enforce the energy-flux condition, not a second observable speed.

## Compactification and continuation

The semi-infinite domain is mapped with

$$x=-L_0\ln(1-s),\qquad 0\le s\le1-\epsilon_{\rm tail}.$$

Solutions must be stable under changes of $L_0$ and $\epsilon_{\rm tail}$. The direct requested interface fraction is tried first; on failure, continuation begins from a smaller fraction and each converged profile seeds the next step. Negative temperatures, nonpositive densities, failed EOS closures and nonphysical transport coefficients are rejected.

This is the manuscript-level method. Current implementation-specific continuation, low-temperature and tail behavior live under [[phase-velocity-overview]], [[unmax-low-temperature]], [[unmax-degeneracy]] and [[thermal-conducting]].

## ⚠ Composition normalization drifts across the notes

The active manuscript defines $a=n_K/n_B$. Parts of `Notes.tex` instead use $(n_K-n_K^Q)/n_B^Q$. Here $n_K^Q=0$ and $n_B$ varies only at the $10^{-10}$ level across the quark layer, so the numerical difference is tiny, but equations should not silently mix the definitions. Use the active-manuscript definition unless documenting a specific solver payload.

## Sources

- `Latex_writting/Hydrodynamical combustion/Main.tex`, first active document: “Hydrodynamic equations,” “Boundary conditions,” and “Numerical method.”
- `Latex_writting/Hydrodynamical combustion/Notes.tex`: “Numerically solving the Energy conserving set of equations” and the cleaner reduced-BVP block. Older entropy-flux formulations are historical and should not be merged into this method.

