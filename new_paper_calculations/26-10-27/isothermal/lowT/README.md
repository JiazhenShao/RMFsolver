# Isothermal contours, low temperature (0.01–10 MeV)

Phase 2 of `docs/superpowers/plans/2026-10-27-isothermal-highT-contours.md`. It will use a Cartesian mesh in ($n_B(0^-)/n_0$, $\log_{10}T$), with 31 rows `np.geomspace(0.01, 10, 31)` and 14 levels through $10^6$ m/s.
It reuses the shared engine, the boundaries and the $\xi$ cache in `../`. No gray-region point is evaluated.
Its own plan is written after the high-T phase.
