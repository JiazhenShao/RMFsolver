"""Benchmark assembly and analytical evaluation for the adiabatic xi scan."""
import os, sys, time, json
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np

ROOT = "/Users/janshao/NSM_related"
C_M_S = 299792458.0
DOMAIN_PATH = f"{ROOT}/new_paper_calculations/26-08-21/T0plus0p1-domain.npy"
FULL_NUMERICAL_PATH = f"{ROOT}/new_paper_calculations/26-08-21/T0plus0p1-uNmax-angular-wavefront.npy"
RHO086_SHELL_PATH = f"{ROOT}/new_paper_calculations/26-08-24/rho0p86_full_theta_shell/T0plus0p1-rho0p86-full-theta-shell.npy"
ADAPTIVE_BRACKET_PATH = f"{ROOT}/new_paper_calculations/26-08-24/rho5k_adaptive_bracket/T0plus0p1-rho5k-adaptive-bracket.npy"


def load(path):
    return np.load(path, allow_pickle=True).item()


def benchmark_points():
    """Union of successful numerical records keyed by (theta_index, radial fraction)."""
    points = {}
    sources = (("wavefront", load(FULL_NUMERICAL_PATH)["radial_shell_records"]),
               ("rho0p86_shell", load(RHO086_SHELL_PATH)["records"]),
               ("adaptive_bracket", load(ADAPTIVE_BRACKET_PATH)["records"]))
    for source, records in sources:
        for r in records:
            if not r["success"]:
                continue
            key = f"{int(r['theta_index'])}:{float(r['radial_fraction_used']):.12f}"
            points[key] = dict(key=key, source=source, theta_index=int(r["theta_index"]),
                               radial_fraction=float(r["radial_fraction_used"]),
                               nB_0minus_over_n0=float(r["nB_0minus_over_n0"]),
                               T_0minus_MeV=float(r["T_0minus_MeV"]),
                               u_num_m_s=float(r["u_0minus"]) * C_M_S,
                               a_0plus_num=float(r["a_0plus"]),
                               lambda_n_num=float(r["lambda_n"]))
    return points


_DOMAIN = None


def evaluate(point, xi=-0.85, methods=("analytic", "semi_analytic")):
    """Same call sequence as adiabatic_xi_minus0p85/_adiabatic_physics.evaluate_method."""
    global _DOMAIN
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from RMFsolver import phase_velocity, RMFparameter as para
    if _DOMAIN is None:
        _DOMAIN = load(DOMAIN_PATH)
    domain = _DOMAIN
    out = dict(key=point["key"], xi=float(xi))
    started = time.monotonic()
    try:
        state = phase_velocity._validated_pnm_state_from_nB(
            point["nB_0minus_over_n0"] * domain["calibration"]["n0_MeV3"], point["T_0minus_MeV"],
            param=para.paraQMCRMF3)
        density_error = abs(state["upstream_density_relative_residual"])
        if not np.isfinite(density_error) or density_error > domain["density_match_rtol"]:
            raise RuntimeError(f"Direct upstream density residual {density_error:g}")
    except Exception as exc:
        for m in methods:
            out[m] = dict(success=False, message=f"upstream: {type(exc).__name__}: {exc}")
        return out
    for m in methods:
        function = dict(analytic=phase_velocity.analytic_velocity_bound,
                        semi_analytic=phase_velocity.semi_analytic_velocity_bound)[m]
        t0 = time.monotonic()
        try:
            res = function(state["muB_0minus"], point["T_0minus_MeV"], domain["calibration"]["B14_MeV"],
                           T_0plus=0.1, xi=float(xi), ms=domain["ms_MeV"], param=para.paraQMCRMF3,
                           NM_type=domain["NM_type"], upB=domain["upB"])
            ok = bool(res.get("success", True))
            row = dict(success=ok, message=str(res.get("message", "")), runtime_s=time.monotonic() - t0)
            for k in ("u_0minus", "jB", "a_0plus", "A_boundary", "lambda_n", "I2_numerical",
                      "one_plus_xi_a_0plus", "muB_inf", "T_inf", "residual"):
                v = res.get(k, np.nan)
                try:
                    row[k] = float(v)
                except (TypeError, ValueError):
                    row[k] = np.nan
            if ok and not (np.isfinite(row["u_0minus"]) and row["u_0minus"] > 0):
                row.update(success=False, message="non-positive or non-finite u")
        except Exception as exc:
            row = dict(success=False, message=f"{type(exc).__name__}: {exc}", runtime_s=time.monotonic() - t0)
        out[m] = row
    out["runtime_s"] = time.monotonic() - started
    return out
