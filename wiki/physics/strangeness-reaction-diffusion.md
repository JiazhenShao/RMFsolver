---
summary: The front is driven by d+u<->u+s weak relaxation and flavor diffusion; $\alpha_s=0.3$ is a phenomenological bag-model input, not controlled running at 400 MeV.
status: current
updated: 2026-08-14
tags: [physics, weak-rates, diffusion, transport]
---

# Strangeness reaction and diffusion

The microscopic competition that sets the laminar front speed is:

- nonleptonic weak relaxation $d+u\leftrightarrow u+s$, which removes positive $\mu_K$;
- strong-interaction diffusion of the down--strange imbalance back toward the interface.

## Weak source

For positive $\mu_K$, the active manuscript uses the Madsen form

$$\Gamma_K=\mathcal A_K\mu_u^5\mu_K(\mu_K^2+4\pi^2T^2),$$

$$\mathcal A_K=\frac{16}{5\pi^5}G_F^2\sin^2\theta_C\cos^2\theta_C.$$

In the quadratic [[quark-matter-eos|EOS]] approximation this becomes

$$\Gamma_K=n_B\gamma_K(a^3+\eta a),$$

$$\gamma_K=\frac{128}{27\times5\pi^3}G_F^2\sin^2\theta_C\cos^2\theta_C\,\mu_q^5,
\qquad \eta=\frac{9\pi^2T^2}{\mu_q^2}.$$

The cubic term is suprathermal; the term linear in $a$ is the finite-temperature subthermal contribution. The analytic formula uses this expansion, whereas the numerical calculation is intended to use the full thermodynamic rate.

## Flavor diffusion

For unpaired degenerate quark matter,

$$D_K^{-1}=\frac{8N_f\alpha_s^2}{\pi}
\left(h_D\frac{T^{5/3}}{q_D^{2/3}}+\frac{\pi^3T^2}{12q_D}\right),$$

with $N_f=3$, $\alpha_s=0.3$, $h_D\simeq1.81$ and

$$q_D^2=\frac{g^2N_f\mu_q^2}{2\pi^2},\qquad g^2=4\pi\alpha_s.$$

The $T^{5/3}$ term comes from Landau-damped transverse gluons and dominates the analytic low-temperature treatment. The full numerical calculation keeps both displayed terms because they become comparable in the merger-temperature range.

The fixed value $\alpha_s=0.3$ is a conventional phenomenological choice in bag-model studies, represented by the parameter survey of Farhi and Jaffe and the review by Madsen. It should not be described as a controlled value obtained by running the vacuum QCD coupling directly to a scale of 400 MeV; modern perturbative calculations find reasonable convergence only at substantially higher quark chemical potentials.

This $D_K$ is **flavor diffusion**, not thermal diffusivity. The two relaxation channels do not share a common time; see [[quark-transport]].

## Modeling limits

- The nonleptonic channel is the fastest flavor process in the stated regime. Urca channels are omitted and generally add slower equilibration requirements, so the computed laminar speed is best interpreted as an upper bound.
- The single-$D_K$ K equation follows from neglecting cross-diffusion terms and taking the down and strange diagonal diffusion coefficients equal. The older `Notes.tex` derivation flags a missing independent flavor combination; this reduction is an assumption, not a general three-component transport theorem.
- Pairing changes both the weak rates and transport and is outside the model.
- `Notes.tex` contains an order-of-magnitude argument that neutrinos neither trap nor cool efficiently across the microscopic layer, but the active manuscript still lists neutrino transport as future work. Treat neutrino irrelevance as plausible, not settled.
- The [[lte-composition-bound]] is needed before taking $T(0^+)\to0$, where $D_K$ and the mean free path diverge.

## ⚠ Exact-rate chemical potential

The active manuscript alternates between $\mu_u=\mu_B/3$ and an exact code path using $\mu_Q$ from charge neutrality. This is the same unresolved model mismatch recorded in [[quark-matter-eos]] and must be fixed before quoting the “exact” numerical rate as matching the written EOS.

## Sources

- `Latex_writting/Hydrodynamical combustion/Main.tex`, first active document: “Strangeness creation” and “Strangeness diffusion.”
- `Latex_writting/Hydrodynamical combustion/Notes.tex`: “Basis Change from Flavors to Kaoness” and the neutrino discussion; these are working notes rather than final manuscript claims.
- Farhi and Jaffe, *Phys. Rev. D* **30**, 2379 (1984), for the classic bag-model survey including $\alpha_s=0.3$.
- Madsen, *Lect. Notes Phys.* **516**, 162 (1999), arXiv:astro-ph/9809032, for the standard strange-quark-matter review.
