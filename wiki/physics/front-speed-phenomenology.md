---
summary: At fixed $T(0^+)=5$ MeV, the full numerical speeds span tens to thousands of m/s, temperature shifts the zero-speed boundary, and fitted 1-km trajectories catch it in 38--107 s.
status: current
updated: 2026-08-17
tags: [physics, front-speed, phenomenology, results]
---

# Front-speed phenomenology and confidence levels

⚠ At $n_B=3n_{\rm sat}$, $B^{1/4}=180$ MeV and QMC-RMF3, this page previously quoted `analytic_velocity_bound` reference points of 196, 202 and 228 m/s at upstream temperatures 5, 10 and 20 MeV. **Those numbers do not reproduce under the current API** (re-measured 2026-08-17). The full analytical bound at $T(0^+)=5$ MeV gives 143.93, 147.82, 163.23 m/s; the LTE variant, the closest path, gives 189.22, 193.84, 212.06. The retired triple predates an unidentified change in the analytic formulation. Full table and the exact calls in [[phase-velocity-overview]].

These remain useful API-level checks, but they do not describe the full fixed-interface-temperature scan.

Over a 10 ms postmerger emission window, even $10^3$ m/s advances a front by only 10 m. That is negligible beside kilometer-sized critical-pressure regions. Existing instantaneous-conversion simulations therefore require an effective propagation rate roughly $300$--$10^4$ times faster than the laminar reaction--diffusion model; see [[gw-observables-section]].

## Full numerical fixed-$T(0^+)$ result

The August 13 numerical contour fixes $T(0^+)=5$ MeV, uses QMC-RMF3 pure neutron matter with $m_s=0$ and $\alpha_s=0.3$, and calibrates $B^{1/4}=189.1566$ MeV so that cold coexistence occurs at $3n_0$. The accepted cells span approximately 25--3415 m/s. This is a conditional planar, laminar result: prescribing the interface temperature does not supply a thermal interface closure.

The allowed branch lies between the zero-speed boundary, where the quark-side profile collapses, and the excluded $A=1$ boundary. Increasing density away from the zero-speed boundary generally raises the speed. Temperature does not produce a single monotonic diffusion-coefficient trend because it also changes the EOS trajectory, the exact weak rate, and the location of the zero-speed boundary. That boundary moves from $n_{B,0}/n_0=3.013$ at $T(0^-)=0$ MeV to 2.910, 2.542 and 1.413 at 20, 40 and 60 MeV.

For the plotted propagation exercise, $z$ is the remaining distance to the temperature-dependent zero-speed surface and $n_B(0^-;z)/n_0=n_{B,0}(T(0^-))/n_0+(0.3\,{\rm km}^{-1})z$. Starting from $z=1$ km, the fitted catch-up times are 38.08, 38.19, 43.19 and 107.19 s for $T(0^-)=0$, 20, 40 and 60 MeV. The fitted terminal powers are all below one, so all four curves reach $z=0$ in finite time within this closure.

Most of each duration is accumulated in the fitted low-speed tail. The resolved/tail contributions are 15.2/22.8, 15.4/22.8, 16.2/27.0 and 11.5/95.7 s, respectively. Tail-window sensitivity raises the upper estimates to about 42.7, 42.8, 48.5 and 148.9 s, making the 60 MeV case especially uncertain.

## What the analytic model explains

- The speed is set parametrically by a weak-rate/diffusion competition, schematically $u_N\sim\sqrt{\gamma_KD_K}$ times composition and hydrodynamic factors.
- The formal speed grows as the newborn quark composition approaches the neutron flavor content, because less weak equilibration is required.
- The formal $A\to1$ divergence is a breakdown of the diffusion-limited description, not an infinite physical velocity. For fixed $A<1$, [[lte-composition-bound]] excludes the exact $a(0^+)=A$ endpoint, but it does **not** prove that the $A\to1$ divergence is regularized; that requires a resolved [[interface-closure]] or a separate deflagration-breakdown argument.
- The equilibrium limit should have vanishing propagation speed as the nuclear metastability goes to zero, but the production contour realizing this must use the temperature-dependent critical state consistently.

## Superseded draft claims

The inherited Results text described a broad $10^2$--$10^3$ m/s band, a mostly decreasing speed with temperature, and higher-temperature $z(t)$ curves that never reach the zero-speed surface. The August 13 numerical contour and tail analysis supersede all three statements. Its separate interface-mode estimates of 1--15 m/s in inspiral and order $10^3$ m/s postmerger have not been re-evaluated here.

The sources of the older mismatch were:

- several visible/hidden contours use or discuss the superseded $T(0^+)=0$ maximization;
- the old numerical uNmax branch is nonunique at exactly zero interface temperature ([[unmax-degeneracy]]);
- a one-factor $D_K\propto T^{-5/3}$ argument omitted the temperature dependence of the zero-speed boundary, EOS trajectory and exact weak rate;
- the definition of metastability and the tuned-versus-fixed bag parameterization changed during the draft.

## Extensions that may lower or restructure the laminar result

Realistic beta-equilibrated nuclear matter, additional Urca-controlled fractions, pairing, neutrino transport, upstream disturbances, dissipative stresses, time dependence, reverse conversion and phase-conversion dissipation all lie outside the baseline model. Some plausibly slow the front, but their net direction is not established by the present calculation. Turbulence, front instabilities or dense nucleation can instead increase the **effective volume-conversion rate** without changing the calculated planar laminar speed; they must not be folded into $u_N$ silently.

## Sources

- [[phase-velocity-overview]] for current reference evaluations.
- `Latex_writting/Hydrodynamical combustion/Main.tex`, first active document: current “Results” and “Conclusion.”
- `new_paper_calculations/26-08-12/T0plus5-all-zero-upstream-v1-uNmax-radial-continuation-T0minus0-local-patched.npy` for the fixed-$T(0^+)$ contour.
- `26SU-12.ipynb` and `local_analysis_results/26SU-12/full-numerical-z-time-evolution.pdf` for the propagation exercise and terminal fit.
