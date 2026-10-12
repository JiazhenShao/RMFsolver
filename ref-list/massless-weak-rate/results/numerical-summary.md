# Numerical validation

The reference is the massless four-quark channel only, with GF = 1.16637e-11 MeV^-2 and theta_C = 13.02 degrees.
The neutral EOS uses net massless-quark densities n_i = mu_i^3/pi^2 + mu_i*T^2; this does not add antiquark reaction channels.
The fixed-muB,a grid holds the local composition exactly. The fixed-muB,muK grid fixes muK to its cold value at the listed a_cold.
All approximation formulas in the grid are evaluated at the same finite-temperature EOS chemical potentials as the reference.

## Verification

- cold_max_2d_analytic_relative_error: 5.14033e-14
- cold_max_3d_analytic_relative_error: 1.71609e-08
- finite_3d_max_relative_error_n24: 3.10862e-15
- detailed_balance_relative_error: -2.22045e-16
- equilibrium_rate: 0
- sign_reversal_relative_error: 0
- dimensional_scaling_relative_error: 0
- eos_path_C2_max_corrected_relative_error: 9.70892e-08
- grid_max_relative_n24_to_n40: 1.70752e-13
- grid_max_relative_tail48_to_tail64: 5.55112e-15
- wall_seconds: 195.884
- Largest relative change from 24 to 40 nodes per subinterval: 1.70752e-13.
- Largest relative tail change from 48 T to 64 T: 5.55112e-15.

## Representative errors at fixed composition

Errors are (approximation / exact phase-space integral - 1) * 100 percent.
Every figure compares only the exact phase-space integral, Madsen Eq. (23), and the derived T0+T2 rate.
The exact reference has zero error; extra T3 and Eq. (24) diagnostics remain in the CSV but are not plotted.

| muB | a | T | Exact phase space | Madsen Eq. (23) | Derived T0+T2 |
|---|---|---|---|---|---|---|
| 900 | 0.146 | 1 | 0% | 7.37% | 0.0126% |
| 900 | 0.146 | 10 | 0% | 15.5% | 5.36% |
| 900 | 0.146 | 50 | 0% | 31.8% | 17.9% |
| 900 | 0.146 | 100 | 0% | -15.8% | -27.2% |
| 900 | 0.9 | 1 | 0% | 178% | 0.000252% |
| 900 | 0.9 | 10 | 0% | 178% | 0.142% |
| 900 | 0.9 | 50 | 0% | 168% | -7.1% |
| 900 | 0.9 | 100 | 0% | 73.9% | -38.1% |
| 900 | 1 | 1 | 0% | 812% | 0.000126% |
| 900 | 1 | 10 | 0% | 785% | 0.0881% |
| 900 | 1 | 50 | 0% | 451% | -2.91% |
| 900 | 1 | 100 | 0% | 148% | -31.4% |
| 1200 | 0.146 | 1 | 0% | 7.31% | 0.00522% |
| 1200 | 0.146 | 10 | 0% | 12.7% | 3.15% |
| 1200 | 0.146 | 50 | 0% | 34.4% | 21.1% |
| 1200 | 0.146 | 100 | 0% | 11.2% | -2.17% |
| 1200 | 0.9 | 1 | 0% | 178% | 0.000107% |
| 1200 | 0.9 | 10 | 0% | 178% | 0.0729% |
| 1200 | 0.9 | 50 | 0% | 177% | -2.18% |
| 1200 | 0.9 | 100 | 0% | 128% | -21.7% |
| 1200 | 1 | 1 | 0% | 812% | 5.37e-05% |
| 1200 | 1 | 10 | 0% | 796% | 0.0418% |
| 1200 | 1 | 50 | 0% | 559% | -0.211% |
| 1200 | 1 | 100 | 0% | 274% | -14.8% |

## Scope and limitations

The T2 coefficient is the fixed-chemical-potential coefficient. Evaluating I0(mu(T)) + C2(mu(T))*T^2 automatically includes the EOS displacement but is not identical to re-expanding at T=0.
Coincident incoming/outgoing up-quark surfaces produce a T3 correction. The saved residual sweep tests this directly; a T4-only remainder claim would be incorrect.
The a=1 endpoint has mu_s=0. The coefficient formulas have finite limits there, but ordinary strange-surface degeneracy assumptions do not hold.
These are rate errors for local thermodynamic states, not errors in a solved front velocity. Large-temperature extrapolations of truncated series are diagnostics only.
The quadrature reports convergence differences, not rigorous interval-arithmetic error bounds.
