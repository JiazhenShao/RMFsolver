# 26SU-3 result summary

Ran 16/16 planned temperature/metastability trials.
Successful solver profiles: 15; failed solver profiles: 1.

Failed points:
- T=20 MeV, M=0.25: Initial energy-conserving state construction failed: Energy-conserving local quark-state solve failed: single-guess energy-conserving quark-

Variables passing the strict 1e-06 relative-range test in every successful trial: E_reconstructed, h_profile, nB, r_gamma_profile, u
Variables with at least one non-constant failure above 0.0001: D_profile, eta_profile, gamma_profile, muB, q
Variables only in the near band (1e-06, 0.0001]: none

Worst relative ranges:
- q: 1.000e+00
- eta_profile: 8.718e-01
- D_profile: 8.385e-01
- gamma_profile: 2.688e-03
- muB: 5.382e-04
- h_profile: 2.441e-12
- nB: 1.268e-12
- u: 1.268e-12
- E_reconstructed: 1.177e-12
- r_gamma_profile: 0.000e+00

The reconstructed conserved energy flux E = h_profile * u * r_gamma_profile has worst relative range 1.177e-12 across successful trials.
