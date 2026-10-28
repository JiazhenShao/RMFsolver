"""One analytic or numerical evaluation at (x = n_B(0-)/n_0, T(0-)), with
method-aware acceptance and the intermediates saved for later xi retuning.

Analytic speed: v(xi) = c*sqrt(2*I2/(lambda_n^2*(1-a)*(1+xi*a))), with I2,
lambda_n and a = a(0+) independent of xi and stored on every record.
Numerical records add the manuscript profile integrals I1 and I2 and the
profile xi they imply, plus a 256-node profile for later diagnostics.
"""
import numpy as np

from RMFsolver import phase_velocity as pv
import _iso_common as common

ROUNDOFF_STATUSES = ("stable_neutron_matter", "isothermal_coexistence",
                     "composition_ceiling_saturated")
ROUNDOFF_MESSAGES = ("Automatic interface ceiling is not interior",)
TABLE_FAILURE = "Slow-front EOS/diffusion table did not converge"
TABLE_LADDER = (0.5, 0.2, 0.1)   # path_rtol factors, as in the 26-10-24 adiabatic wrapper
ENDPOINTS = ("P_0minus", "muB_0minus", "nB_0minus", "muB_0plus", "muK_0plus", "nB_0plus",
             "muB_inf", "nB_inf")
ANALYTIC_FIELDS = ("I2", "lambda_n", "eta", "D_K", "invD_K", "gamma_K", "qD", "delta_muB")
NUMERICAL_FIELDS = ("solution_method", "initial_guess_used", "bvp_status", "bvp_max_rms_residual",
                    "shooting_flux_change", "shooting_mismatch", "closure_error_max",
                    "lambda_inf", "jB", "u_0minus", "bvp_nodes")
PROFILE_FIELDS = ("x", "a", "nK", "jK", "nB", "D_K", "Gamma_K", "muB", "muK")


def _base(method, x, T):
    return dict(method=method, x=float(x), T=float(T), status="unresolved", accepted=False,
                speed_m_s=np.nan, message="", attempts=0, a_0plus=np.nan)


def _copy(record, result, keys):
    for key in keys:
        value = result.get(key, np.nan)
        record[key] = value if isinstance(value, str) or value is None else float(value)


def _analytic(x, T):
    record = _base("analytic", x, T)
    record["attempts"] = 1
    try:
        result = pv.analytic_velocity_isothermal_slow(
            T, x * common.N0_MEV3, common.PHYSICS["B_one_forth"], xi=common.XI)
    except (RuntimeError, ValueError, FloatingPointError) as exc:
        return dict(record, message=str(exc)), None
    record["message"] = result.get("message", "")
    if result.get("status") in ROUNDOFF_STATUSES:
        return dict(record, status="boundary_roundoff"), None
    if not (result.get("success") and result.get("front_exists")):
        return record, None
    _copy(record, result, ENDPOINTS + ANALYTIC_FIELDS + ("a_0plus",))
    record.update(status="accepted", accepted=True, speed_m_s=float(result["velocity_m_s"]),
                  xi_used=float(result["xi"]))
    return record, None


def numerical_acceptance(result, tol_bvp):
    """Reason the solver result is not acceptable, or '' when it is."""
    if not result.get("success"):
        return result.get("message") or "solver failure"
    speed, a = result.get("u_0minus", np.nan), result.get("a_0plus", np.nan)
    if not (np.isfinite(speed) and speed > 0 and 0 < a < 1):
        return "nonphysical speed or composition"
    # Every metric must be finite and within tolerance: max() and '>' both let NaN through.
    within = lambda values, tolerance: (len(values) > 0 and bool(np.all(np.isfinite(values)))
                                        and bool(np.all(values <= tolerance)))
    residuals = np.array([result.get(k, np.nan) for k in ("constitutive_residual_norm",
                          "reaction_residual_norm", "diffusion_integral_residual_norm")], dtype=float)
    if not (within(np.array([result.get("closure_error_max", np.nan)], dtype=float), 1e-8)
            and within(np.array(list(result.get("conservation_errors", {}).values()), dtype=float), 1e-8)
            and within(np.abs(np.asarray(result.get("boundary_residuals", []), dtype=float)), 1e-7)
            and within(residuals, max(5 * tol_bvp, 1e-7))):
        return "direct physical validation"
    if result.get("solution_method") == "shooting":
        change, mismatch = result.get("shooting_flux_change", np.nan), result.get("shooting_mismatch", np.nan)
        if not (np.isfinite(change) and change <= 1e-6 and np.isfinite(mismatch) and abs(mismatch) <= 1e-8):
            return "shooting flux change or mismatch"
        return ""
    rms = result.get("bvp_max_rms_residual", np.nan)
    if result.get("bvp_status") != 0 or not np.isfinite(rms) or rms > tol_bvp:
        return "collocation status or nonfinite RMS"
    return ""


def _profile_integrals(record, result):
    """Manuscript I1, I2 on the numerical profile; D_K a' = (jB a - J_K)/n_B (a' = n_K'/n_B)."""
    x, a, jK, nB = (np.asarray(result[k], dtype=float) for k in ("x", "a", "jK", "nB"))
    D_K, Gamma = np.asarray(result["D_K"], dtype=float), np.asarray(result["Gamma_K"], dtype=float)
    jB = float(result["jB"])
    flux = (jB * a - jK) / nB
    I1 = float(np.trapezoid(flux**2 / D_K, x))
    I2 = float(-np.trapezoid(Gamma * flux, x) / nB[0])
    a0, u = float(result["a_0plus"]), float(result["u_0minus"])
    for suffix, density in (("", result["nB_0plus"]), ("_inf", result["nB_inf"])):
        lam = float(result["nB_0minus"]) / float(density)
        record["xi_profile" + suffix] = 2 * I1 / (lam * u * (1 - a0) * a0) - 1
    record.update(I1_num=I1, I2_num=I2)


def _resampled_profile(result):
    s = np.asarray(result["s_coord"], dtype=float)
    grid = np.linspace(s[0], s[-1], 256)
    return {key: np.interp(grid, s, np.asarray(result[key], dtype=float)) for key in PROFILE_FIELDS}


def _numerical(x, T, overrides=None):
    record = _base("numerical", x, T)
    settings = dict(common.NUMERICAL, **(overrides or {}))
    base_path_rtol = settings["path_rtol"]
    ladder, mesh_retry_used, reason = iter(TABLE_LADDER), False, ""
    while True:
        record["attempts"] += 1
        try:
            result = pv.solve_front_isothermal_slow(
                T, x * common.N0_MEV3, common.PHYSICS["B_one_forth"], return_profile=True, **settings)
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            result = dict(success=False, message=str(exc))
        message = result.get("message", "")
        if message in ROUNDOFF_MESSAGES:
            record.update(status="boundary_roundoff", message=message,
                          a_0plus_max_status=result.get("a_0plus_max_status"))
            return record, None
        reason = numerical_acceptance(result, settings["tol_bvp"])
        if not reason:
            break
        record["message"] = message
        # The table-refinement failure walks the stricter-tolerance ladder of the
        # adiabatic contour wrapper; any other failure gets one 600-node retry.
        factor = next(ladder, None) if message == TABLE_FAILURE else None
        if factor is not None:
            settings["path_rtol"] = base_path_rtol * factor
        elif message != TABLE_FAILURE and not mesh_retry_used:
            mesh_retry_used, settings["n_mesh"] = True, 600
        else:
            break
    if reason:
        record.update(message=record["message"] or reason, rejection=reason)
        return record, None
    _copy(record, result, ENDPOINTS + NUMERICAL_FIELDS + ("a_0plus",))
    record.update(status="accepted", accepted=True, message=message,
                  speed_m_s=float(result["u_0minus"]) * common.C_M_S,
                  conservation_max=float(max(result["conservation_errors"].values())),
                  n_mesh_used=settings["n_mesh"], path_rtol_used=settings["path_rtol"])
    _profile_integrals(record, result)
    return record, _resampled_profile(result)


# path_rtol=1e-7: next to the udd boundary the backward-shot flux is reproducible only to
# ~1e-6 with the default tables, the size of its own self-check; tighter tables lower that floor.
REPAIR_VARIANTS = (dict(initial_guess_mode="shooting", path_rtol=2e-7), dict(tol_bvp=2e-5), dict(n_mesh=1200),
                   dict(path_rtol=1e-7))


def evaluate_repair(method, x, T):
    """Repair passes only: the standard attempts, then documented solver variants in order.

    Same acceptance rules; the record names the variant that succeeded.
    """
    record, profile = evaluate(method, x, T)
    if record["accepted"] or method != "numerical" or record["status"] == "boundary_roundoff":
        return record, profile
    attempts = record["attempts"]
    for variant in REPAIR_VARIANTS:
        trial, trial_profile = _numerical(float(x), float(T), variant)
        attempts += trial["attempts"]
        if trial["accepted"]:
            return dict(trial, attempts=attempts, repair_variant=variant), trial_profile
    return dict(record, attempts=attempts, repair_variants_tried=list(REPAIR_VARIANTS)), None


def evaluate(method, x, T):
    """(record, profile-or-None) for method in {'analytic', 'numerical'}."""
    if method == "analytic":
        return _analytic(float(x), float(T))
    if method == "numerical":
        return _numerical(float(x), float(T))
    raise ValueError("method must be 'analytic' or 'numerical'")
