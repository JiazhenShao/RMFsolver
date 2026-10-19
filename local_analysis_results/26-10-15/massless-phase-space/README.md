# Massless weak phase-space benchmarks

Calculated on 2026-10-15 for independent collaborator checks.
These are fresh evaluations of the full finite-temperature four-quark integral, not values digitized from a figure or copied from the manuscript CSV.

## Data

- `phase-space_all.csv`: all 60 states, one state per row.
- `phase-space_muB-*_a-*.csv`: six individual panels, ten temperatures each.
- `convergence.csv`: quadrature, tail, independent-integral and EOS checks for every state. Empty entries mean that a particular check was not performed at that state.
- `metadata.json`: constants, numerical settings, verification results, software versions and source SHA-256 hashes.
- `calculate.py` and `reference/`: reproducible calculation and snapshots of the exact source modules used.

The six panels use mu_B = 900,1200 MeV and a = 0.146,0.5,0.9.
Temperatures are 0,0.01,0.1,1,5,10,20,25,50,100 MeV.
The screenshot uses the eight positive-temperature values excluding 20 MeV; 0 and 20 MeV are additional benchmarks.
The T=0 row is a direct cold phase-space calculation, not a small-temperature substitution. Omit that row on a logarithmic temperature axis.

## State definition

Each temperature holds mu_B and the local neutral composition a fixed, while recomputing the flavor potentials using the exact massless free-gas net-density EOS:

```text
n_f = mu_f^3/pi^2 + mu_f*T^2
mu_B = mu_u + mu_d + mu_s
n_B = (n_u + n_d + n_s)/3
2*n_u = n_d + n_s
a = (n_d - n_s)/(2*n_B)
mu_K = mu_d - mu_s
```

Thus n_u=n_B, n_d=(1+a)*n_B and n_s=(1-a)*n_B.
Use the supplied mu_u, mu_d and mu_s to compare collision integrals directly, without introducing differences from another EOS inversion.
This is the fixed-mu_B, fixed-a comparison shown in the image, not the conserved adiabatic trajectory calculation.

## What the integral columns mean

All energies and momenta are in MeV; hbar=c=k_B=1.
The forward reaction is d+u -> s+u, the reverse is s+u -> d+u, and the net rate is positive for the states in this file.
The unprefactored phase-space normalization is exactly:

```text
f_f(p) = 1/(1 + exp((p-mu_f)/T))
L = (E-k)/2
H = (E+k)/2

phase_space_forward = integral(E=0..infinity) dE integral(k=0..E) dk
  (E^2-k^2)^2
  [integral(p=L..H) dp f_u(p)*f_d(E-p)]
  [integral(p=L..H) dp (1-f_u(p))*(1-f_s(E-p))]

reverse_to_forward_factor = exp(-mu_K/T)
detailed_balance_factor = 1-exp(-mu_K/T)
phase_space_reverse = reverse_to_forward_factor*phase_space_forward
phase_space_net = detailed_balance_factor*phase_space_forward

weak_prefactor = 9*G_F^2*sin(theta_C)^2*cos(theta_C)^2/(8*pi^5)
rate_forward = weak_prefactor*phase_space_forward
rate_reverse = weak_prefactor*phase_space_reverse
rate_exact_net = weak_prefactor*phase_space_net
```

The two bracketed momentum integrals are evaluated analytically; the remaining two-dimensional energy/momentum integral retains the full Fermi-Dirac distributions.
The phase-space columns have units MeV^8, the weak prefactor MeV^(-4), and all rate columns MeV^4.
The numerical constants are G_F=1.16637e-11 MeV^(-2) and theta_C=13.02 degrees.
The supplied weak-prefactor column includes the normalization used in the manuscript; do not multiply by a further spin/color factor when comparing this convention.

At T=0 the reverse factor is zero and the net factor is one.
At sufficiently small positive T the reverse factor underflows to zero in double precision; `log_reverse_to_forward_factor` retains its finite logarithm. This log column is empty at T=0, where its limit is negative infinity.
The reverse integral is obtained from detailed balance, not a separate reverse-channel quadrature.

## Column definitions

| Column | Meaning |
|---|---|
| `mu_B_MeV`, `a`, `T_MeV` | Three input state coordinates; a is dimensionless. |
| `mu_u_MeV`, `mu_d_MeV`, `mu_s_MeV` | Neutral EOS flavor potentials at that temperature. |
| `mu_K_MeV` | mu_d-mu_s. |
| `n_B_MeV3` | Baryon density in natural units, not fm^(-3). |
| `detailed_balance_factor` | Multiplicative factor converting forward to net phase space. |
| `reverse_to_forward_factor`, `log_reverse_to_forward_factor` | Reverse/forward factor and its natural logarithm. |
| `phase_space_forward_MeV8`, `phase_space_reverse_MeV8`, `phase_space_net_MeV8` | Unprefactored integrals defined above. |
| `weak_prefactor_MeV_minus4` | Multiplicative weak normalization defined above. |
| `rate_forward_MeV4`, `rate_reverse_MeV4`, `rate_exact_net_MeV4` | Corresponding weak rates; the last is the black curve in the image. |
| `rate_Madsen_Eq23_MeV4` | Blue curve: Eq. (23) using the actual mu_u. |
| `rate_Eq23_muB_over_3_MeV4` | Green curve: replace only mu_u^5 by (mu_B/3)^5 in Eq. (23); keep the same mu_K. |
| `rate_derived_T0_T2_MeV4` | Orange curve: exact cold term plus the derived quadratic thermal term, evaluated at the same local flavor potentials. |

For clarity, both Eq. (23) columns use

```text
C_K = 16*G_F^2*sin(theta_C)^2*cos(theta_C)^2/(5*pi^5)
rate_Madsen_Eq23 = C_K*mu_u^5*mu_K*(mu_K^2+4*pi^2*T^2)
rate_Eq23_muB_over_3 = C_K*(mu_B/3)^5*mu_K*(mu_K^2+4*pi^2*T^2)
```

The green column is a separate chemical-potential approximation; it is not the Eq. (24) correction.
The microscopic model is massless, tree-level, unpaired four-quark scattering. Additional antiquark collision channels and QCD corrections to the weak rate are not included. Thermal antiparticles are included in the net-density EOS above.

## Numerical checks and reproduction

The exported forward integral uses 40-point Gauss-Legendre rules per split subinterval and an upper energy cutoff mu_u+mu_d+64*T.
Every state is also recalculated with 24 and 40 points at a 48*T tail, separating quadrature-order and tail changes.
All six T=0 states are compared against the exact analytical cold polynomial.
All twelve T=20,100 MeV states are checked with a separate three-dimensional radial-momentum representation in which the angular-momentum integral is done first, using 24 points per split subinterval and a 64*T tail.
This independent check shares the physical matrix element but uses a different integration order and integrand implementation.
The 48 states in the saved manuscript grid are compared only after the fresh calculation, if that grid is available locally.
The pass threshold is 1e-6 for relative integral/rate differences; EOS constraints are also checked.
The CSV digits preserve floating-point output and do not claim that every displayed digit is physically or numerically significant.

From this directory, using Python 3 with NumPy and SymPy:

```bash
python3 calculate.py
```

Within the original repository, the equivalent command is:

```bash
./bin/python3 local_analysis_results/26-10-15/massless-phase-space/calculate.py
```

The bundled source files are unchanged copies from `ref-list/massless-weak-rate/` as of the calculation date.
They allow reproduction without importing the production combustion solver or having the manuscript's saved data.
