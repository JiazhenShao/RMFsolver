# THERMAL INTERFACE TEMPERATURE CLOSURE — CURRENT AUTHORITATIVE CONCLUSION

**Date:** 2026-07-23

**Status:** Current research conclusion.

**Read this before editing:** `Latex_writting/Hydrodynamical combustion/Main.tex`, `Latex_writting/Hydrodynamical combustion/Notes.tex`, or `RMFsolver/phase_velocity.py`.

## 1. Problem being closed

The known upstream state is `T(0-)`, `nB(0-)` or `muB(0-)`, and a trial `u(0-)`.
The just-downstream state has four unknowns:

```text
muB(0+), muK(0+), T(0+), u(0+).
```

Baryon-number, energy-flux, and momentum-flux conservation provide three equations.
A fourth sharp-interface relation is therefore required.

## 2. Recommended baseline closure

For a thermally transparent microscopic deconfinement layer, impose

```text
T(0+) = T(0-).
```

This is the zero-interfacial-thermal-resistance limit.
It should be described as a locally isothermal or thermally transparent interface, not as an adiabatic interface.
The equality is a constitutive interface assumption supported by a short thermal-equilibration scale; it is not a consequence of the three ideal-hydrodynamic conservation equations alone.

The current scope deliberately leaves the `x > 0` bulk solver unchanged.
The existing conservation equations determine `T(x)` algebraically within the ideal-hydrodynamic approximation, but they do not explicitly contain a conductive flux `-kappa*dT/dx`.

## 3. Thermal diffusivity is not the flavor-diffusion coefficient

Do not use the flavor-diffusion coefficient `D_K` as if it were the thermal diffusivity.
Heiselberg and Pethick calculate different relaxation times for different transport processes.
In the low-temperature degenerate regime, the thermal-relaxation rate has a different temperature dependence from the momentum/flavor transport rate.

For unpaired degenerate quark matter at `T << qD`, Heiselberg–Pethick give

```text
1/tau_kappa
  = (4/pi^3)*Nf*alpha_s^2*mu_q^2 * (2*zeta(3)*T/qD^2),

qD^2 = 2*Nf*alpha_s*mu_q^2/pi.
```

Wolfram simplification gives

```text
1/tau_kappa = 4*zeta(3)*alpha_s*T/pi^2,

chi_Q = kappa_Q/cV_Q = tau_kappa/3
      = pi^2/(12*zeta(3)*alpha_s*T).
```

For `alpha_s = 0.3` and `T = 10 MeV`,

```text
chi_Q = 0.228072 MeV^(-1) = 45.005 fm,
tau_kappa = 0.684216 MeV^(-1) = 135.014 fm.
```

The steady advection–diffusion scaling

```text
chi_Q*T'' - v*T' = 0
```

has thermal length

```text
ell_T = chi_Q/v,

v = u/sqrt(1 + u^2).
```

At `v = 1e-6`, `alpha_s = 0.3`, and `T = 10 MeV`, this gives `ell_T ~= 45 nm`.
For a transparent microscopic interface of thickness `delta_int = 1–10 fm`,

```text
v*delta_int/chi_Q ~= 2e-8–2e-7.
```

This supports a negligible temperature variation across the microscopic layer, conditional on the absence of a large interfacial thermal resistance.

## 4. What bulk conductivity cannot determine

A genuine interface temperature jump requires an interfacial conductance `G_int`:

```text
q_int = G_int * (T(0-) - T(0+)).
```

Wolfram verified the nonnegative conductive entropy production

```text
q_int * (1/T(0+) - 1/T(0-))
  = G_int*(T(0-) - T(0+))^2/(T(0-)*T(0+))
  >= 0.
```

This inequality constrains the sign of dissipation but does not determine the magnitude of the temperature jump.
Bulk `kappa_N` and `kappa_Q` do not determine `G_int`.
The literature search found no dense-matter calculation of `G_int` for a sharp nuclear–quark interface.

Therefore the current modeling hierarchy is:

1. Baseline closure: `T(0+) = T(0-)`.
2. Existing independent LTE-validity bound: cap `a(0+)` and infer a lower bound on `T(0+)`.
3. Future extension: introduce and scan a finite `G_int` only when a defensible microscopic model is available.

## 5. Existing LTE composition bound

The existing code implements

```text
beta_LTE = 5*u_N*lambda_n,

lambda_n = nB_N/nB_Q.
```

The saturated LTE composition is obtained from

```text
(1 - beta_LTE)*a_LTE^2 + beta_LTE*a_LTE - A^2 = 0.
```

The numerically stable positive root used in `RMFsolver/phase_velocity.py` is

```text
a_LTE = 2*A^2/(sqrt(beta_LTE^2 + 4*(1 - beta_LTE)*A^2) + beta_LTE).
```

On the analytic constant-pressure trajectory,

```text
T_Qstar_LTE = T_Q*sqrt(1 - (a_LTE/A)^2).
```

This bound is distinct from the transparent-interface equality.
It can be used to study the signed gap between the known upstream temperature and the minimum allowed interface temperature.

## 6. Code facts

- `RMFsolver.phase_velocity._analytic_aqstar_lte` hard-wires the coefficient `5.0` through `beta_LTE = 5.0*u_N*lambda_n`.
- `RMFsolver.phase_velocity.analytic_velocity_bound(..., interface_fraction_mode="LTE")` solves the LTE-limited analytic eigenvalue problem and returns `T_Q`, `A_boundary`, `a_interface_LTE`, `beta_LTE`, `u_N`, and `lambda_n`.
- The returned dictionary does not currently include `T_Qstar_LTE`; calculate it as `T_Q*sqrt(max(0, 1 - (a_interface_LTE/A_boundary)^2))`.
- `solve_front_energy_conserving_uNmax` already accepts the interface temperature through `TQstar`.

## 7. Primary literature

1. H. Heiselberg and C. Pethick, “Transport and relaxation in degenerate quark plasmas,” Physical Review D 48 (1993), DOI `10.1103/PhysRevD.48.2916`.
   Consensus record: https://consensus.app/papers/transport-and-relaxation-in-degenerate-quark-plasmas-heiselberg-pethick/ffb2a69552625b369ac425cf9ca52dd9/?utm_source=chatgpt

2. I. Shovkovy and P. Ellis, “Thermal conductivity of dense quark matter and cooling of stars,” Physical Review C 66 (2002), DOI `10.1103/PhysRevC.66.015802`.
   This work assumes temperature continuity across a directly contacting nuclear–quark interface.
   Consensus record: https://consensus.app/papers/thermal-conductivity-of-dense-quark-matter-and-cooling-of-shovkovy-ellis/efa704c20e7e5fa180e2b49cd0b494cd/?utm_source=chatgpt

3. T. Y. Zhao and N. Patankar, “The thermodynamic principle determining the interface temperatures during phase change,” International Journal of Heat and Mass Transfer (2021), DOI `10.1016/j.ijheatmasstransfer.2022.123389`.
   This is a general phase-change reference showing why an interface temperature can require an additional constitutive principle.
   Consensus record: https://consensus.app/papers/the-thermodynamic-principle-determining-the-interface-zhao-patankar/366e7063840c5f319b2a67df7d13e23a/?utm_source=chatgpt

