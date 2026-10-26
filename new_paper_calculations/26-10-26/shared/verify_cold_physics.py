#!/usr/bin/env python3
"""Independently verify the unchanged cold analytical slow-front APIs.

Run from the repository root with the isolated SciPy runtime on PYTHONPATH.
The default output is a portable JSON report; no contour engine is imported.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from RMFsolver import constants as const
from RMFsolver import phase_velocity as pv


N0 = 1229360.5098323745
B14 = 189.1565957288247
XI = -0.85
ALPHA_S = 0.4
H_D = 1.8131745180482931
PATH_RTOLS = (1e-6, 5e-7, 2e-7, 1e-7)
STATES = ((0.0, 3.5), (30.0, 3.5), (70.0, 2.5))
METHODS = ("analytic_velocity_adiabatic_slow", "semi_analytic_velocity_adiabatic_slow")
DEFAULT_OUTPUT = ROOT / "local_analysis_results/26-10-26/cold-analytic-physics-verification.json"


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def relative_error(value, reference):
    return float(abs(value - reference) / max(abs(reference), np.finfo(float).tiny))


def weak_gamma(muB):
    return (128 / (135 * np.pi**3) * const.G_Fermi**2
            * np.cos(const.Cabibbo_angle_rad)**2 * np.sin(const.Cabibbo_angle_rad)**2
            * (muB / 3)**5)


def leading_diffusion(muB, temperature):
    qD = muB / 3 * np.sqrt(6 * ALPHA_S / np.pi)
    return np.pi * qD**(2 / 3) / (24 * ALPHA_S**2 * H_D * temperature**(5 / 3))


def local_rate(muB, muK, temperature):
    """Explicit modified-Madsen polynomial, independent of the rate helper."""
    return (16 / (5 * np.pi**5) * const.G_Fermi**2
            * np.cos(const.Cabibbo_angle_rad)**2 * np.sin(const.Cabibbo_angle_rad)**2
            * (muB / 3)**5 * muK * (muK**2 + 4 * np.pi**2 * temperature**2))


def inverse_diffusion(muB, temperature):
    """Explicit full two-term transport law, regular at exactly T=0."""
    qD = muB / 3 * np.sqrt(6 * ALPHA_S / np.pi)
    return (24 * ALPHA_S**2 / np.pi
            * (H_D * temperature**(5 / 3) / qD**(2 / 3)
               + np.pi**3 * temperature**2 / (12 * qD)))


def solve(name, T_0minus, n_multiple, T_0plus=0.0, rtols=PATH_RTOLS):
    attempts = []
    for tolerance in (rtols if name == METHODS[1] else (None,)):
        options = dict(T_0plus=T_0plus, xi=XI, ms=0.0,
                       param=pv.para.paraQMCRMF3, NM_type="PNM")
        if tolerance is not None:
            options["path_rtol"] = tolerance
        started = time.perf_counter()
        result = getattr(pv, name)(T_0minus, n_multiple * N0, B14, **options)
        attempts.append({"path_rtol": tolerance, "success": bool(result["success"]),
                         "status": result["status"], "message": result["message"],
                         "elapsed_seconds": time.perf_counter() - started})
        if result["success"]:
            return result, attempts
        # Tightening helps numerical path-table nonconvergence, not physical rejection.
        if "converg" not in result["message"].lower():
            break
    raise RuntimeError(json.dumps({"method": name, "T_0minus": T_0minus,
                                  "n_multiple": n_multiple, "T_0plus": T_0plus,
                                  "attempts": attempts}))


def common_checks(result):
    maximum = max(result["conservation_errors"].values())
    require(maximum <= 1e-8, "Conservation exceeds 1e-8")
    require(result["front_exists"], "No propagating front")
    require(np.isfinite(result["I2"]) and result["I2"] > 0, "I2 is not positive finite")
    require(np.isfinite(result["velocity_m_s"]) and result["velocity_m_s"] > 0,
            "Speed is not positive finite")
    require(result["T_0plus"] == 0, "Cold endpoint has a temperature floor")
    require(np.isposinf(result["D_K_0plus"]), "Cold local diffusion is not +infinity")
    require(result["invD_K_0plus"] == 0, "Cold inverse diffusion is not exactly zero")
    flux_error = max(relative_error(result["nB_" + suffix] * result["u_" + suffix],
                                    result["jB"]) for suffix in ("0minus", "0plus", "inf"))
    require(flux_error <= 1e-12, "Baryon flux is not conserved")
    return {"I2": float(result["I2"]), "velocity_m_s": float(result["velocity_m_s"]),
            "a_0plus": float(result["a_0plus"]), "T_inf": float(result["T_inf"]),
            "D_K_0plus": "+Infinity", "invD_K_0plus": 0.0,
            "conservation_errors": result["conservation_errors"],
            "max_conservation_error": float(maximum),
            "baryon_flux_relative_error": flux_error}


def full_quadrature(result):
    muB, A, T_inf = result["muB_inf"], result["A_adb"], result["T_inf"]

    def integrand(t):
        # t=(1-a^2/A^2)^(1/6); no endpoint-floor substitution is made.
        fraction = A * np.sqrt(1 - t**6)
        temperature = T_inf * t**3
        jacobian = 3 * A * t**5 / np.sqrt(1 - t**6)
        return (leading_diffusion(muB, temperature) * weak_gamma(muB)
                * (fraction**3 + 81 * np.pi**2 * temperature**2 / muB**2 * fraction)
                * jacobian)

    integral, error = quad(integrand, 0, 1, epsabs=1e-28, epsrel=1e-11)
    difference = relative_error(result["I2"], integral)
    require(difference <= 2e-10, "Closed cold I2 disagrees with independent quadrature")
    return {"I2_quadrature": integral, "quad_error_estimate": error,
            "relative_error": difference, "coordinate": "t=(1-a^2/A_adb^2)^(1/6)"}


def semi_direct_checks(result, T_0minus):
    path = result["thermodynamic_path"]
    H = result["h_0minus"] / result["nB_0minus"]
    pressure_scale = max(abs(result["P_0minus"]), B14**4, 1.0)
    checks = {"pressure": 0.0, "enthalpy_per_baryon": 0.0, "constant_density": 0.0,
              "stored_pressure": 0.0, "stored_density": 0.0,
              "rate_relative_error": 0.0, "inverse_diffusion_relative_error": 0.0}
    for i in range(len(path["T"])):
        muB, muK, temperature = (float(path[key][i]) for key in ("muB", "muK", "T"))
        pressure = float(pv.PQM(muB, muK, B14, temperature, ms=0.0))
        energy = float(pv.edensQM(muB, muK, B14, temperature, ms=0.0))
        density = float(pv.nB_QM(muB, muK, B14, temperature, ms=0.0))
        values = {
            "pressure": abs(pressure - result["P_0minus"]) / pressure_scale,
            "enthalpy_per_baryon": abs((pressure + energy) / density / H - 1),
            "constant_density": abs(density / result["nB_inf"] - 1),
            "stored_pressure": abs(pressure - path["P"][i]) / pressure_scale,
            "stored_density": relative_error(density, path["nB"][i]),
            "rate_relative_error": relative_error(path["Gamma_K"][i], local_rate(muB, muK, temperature)),
            "inverse_diffusion_relative_error": relative_error(path["invD_K"][i], inverse_diffusion(muB, temperature)),
        }
        for key, value in values.items():
            checks[key] = max(checks[key], float(value))
    require(max(checks[key] for key in ("pressure", "enthalpy_per_baryon", "constant_density"))
            <= 1e-8, "Direct full EOS conservation exceeds 1e-8")
    require(max(checks[key] for key in ("stored_pressure", "stored_density", "rate_relative_error",
                                      "inverse_diffusion_relative_error")) <= 1e-12,
            "Stored thermodynamics or transport differs from direct evaluation")
    table = pv._slow_front_path("adiabatic", result["P_0minus"], H, T_0minus,
                               B14, 0.0, None, 0.0, result["path_rtol"])

    def integrand(w):
        state = table.point(w)
        return (local_rate(state["muB"], state["muK"], state["T"])
                * table.primitive(w, 1) / result["nB_inf"]**2)

    integral, error = quad(integrand, 0, 1, points=[table.core_join], epsabs=1e-25, epsrel=2e-7)
    difference = relative_error(result["I2"], integral)
    require(difference <= 3e-7, "Semi-analytical I2 disagrees with direct-rate chart quadrature")
    return {"path_samples": len(path["T"]), "direct_checks": checks,
            "I2_quadrature": integral, "quad_error_estimate": error,
            "I2_relative_error": difference, "path_nodes": int(result["path_nodes"]),
            "path_error": float(result["path_error"]),
            "independence_limit": "Direct rate quadrature reuses the existing diffusion primitive; it bypasses the stored rate spline."}


def verify_state(T_0minus, n_multiple):
    record = {"T_0minus_MeV": T_0minus, "nB_0minus_over_n0": n_multiple, "methods": {}}
    for name in METHODS:
        cold, attempts = solve(name, T_0minus, n_multiple)
        summary = common_checks(cold)
        summary["cold_attempts"] = attempts
        summary["integral_check"] = (full_quadrature(cold) if name == METHODS[0]
                                      else semi_direct_checks(cold, T_0minus))
        if name == METHODS[1]:
            tighter = tuple(value for value in PATH_RTOLS if value < cold["path_rtol"])
            require(tighter, "No tighter configured path tolerance remains for refinement")
            refined, refined_attempts = solve(name, T_0minus, n_multiple, rtols=tighter)
            refinement_error = relative_error(refined["velocity_m_s"], cold["velocity_m_s"])
            require(refinement_error <= 1e-4, "Tighter path changes speed by more than 1e-4")
            summary["refinement"] = {"attempts": refined_attempts,
                                     "path_rtol": float(refined["path_rtol"]),
                                     "relative_speed_change": refinement_error,
                                     "refined_velocity_m_s": float(refined["velocity_m_s"]),
                                     "refined_max_conservation_error": max(refined["conservation_errors"].values())}
        convergence = []
        for warm_temperature in (0.01, 1e-5):
            warm, warm_attempts = solve(name, T_0minus, n_multiple, T_0plus=warm_temperature)
            convergence.append({"T_0plus_MeV": warm_temperature, "attempts": warm_attempts,
                                "velocity_m_s": float(warm["velocity_m_s"]),
                                "relative_speed_difference_from_cold": relative_error(warm["velocity_m_s"], cold["velocity_m_s"]),
                                "max_conservation_error": max(warm["conservation_errors"].values())})
        errors = [entry["relative_speed_difference_from_cold"] for entry in convergence]
        require(errors[1] < errors[0] and errors[1] < 0.01, "Warm interface does not converge to the cold integral")
        summary["warm_to_cold"] = convergence
        summary["warm_to_cold_error_ratio"] = errors[1] / errors[0]
        record["methods"][name] = summary
    record["passed"] = True
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    started = time.perf_counter()
    report = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "physics": {"T_0plus_MeV": 0.0, "xi": XI, "B_one_forth_MeV": B14,
                    "n0_MeV3": N0, "ms_MeV": 0.0, "NM_type": "PNM", "parameter_set": "QMC-RMF3"},
        "tolerances": {"conservation": 1e-8, "refinement_speed_relative": 1e-4,
                       "path_rtol_retry_sequence": list(PATH_RTOLS)},
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (Path(__file__).resolve(), ROOT / "RMFsolver/phase_velocity.py",
                                       ROOT / "tests/test_phase_velocity_analytic_slow.py")},
        "limitations": ["This validates representative physical states, not every contour point.",
                        "Warm-to-cold convergence is fractional-power; no tiny equality tolerance is imposed.",
                        "The rate law is the modified-Madsen approximation, not a phase-space integration."],
        "states": [], "passed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        report["elapsed_seconds"] = time.perf_counter() - started
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    try:
        command = [sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests",
                   "-p", "test_phase_velocity_analytic_slow.py", "-v"]
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
        report["regression"] = {"command": command, "returncode": completed.returncode,
                                "stdout": completed.stdout, "stderr": completed.stderr}
        save()
        require(completed.returncode == 0, "Analytical slow-front regression failed")
        for T_0minus, n_multiple in STATES:
            print(f"Verifying T_0minus={T_0minus:g} MeV, nB_0minus/n0={n_multiple:g}", flush=True)
            report["states"].append(verify_state(T_0minus, n_multiple))
            save()
        report["passed"] = True
    except Exception as exc:
        report["failure"] = {"type": type(exc).__name__, "message": str(exc),
                             "traceback": traceback.format_exc()}
    save()
    print(json.dumps({"passed": report["passed"], "states": len(report["states"]),
                      "output": str(args.output), "elapsed_seconds": report["elapsed_seconds"]}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
