"""Calculate reusable phase-transition and entropy-equality plot overlays."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, root

import RMFsolver.RMFparameter as para
from RMFsolver import phase_velocity


def _finite_pressure_residual(muB, temperature, B14, ms, upB, nm_type):
    try:
        return float(
            phase_velocity.PNM(
                float(muB),
                float(temperature),
                param=para.paraQMCRMF3,
                NM_type=str(nm_type),
            )
            - phase_velocity.PQM(
                float(muB),
                0.0,
                float(B14),
                float(temperature),
                ms=float(ms),
                upB=int(upB),
            )
        )
    except Exception:
        return np.nan


def _uds_transition_curve(
    temperatures,
    *,
    n0,
    B14,
    muB_crit0,
    ms,
    upB,
    nm_type,
    muB_limits,
    bracket_half_widths,
    fallback_probe_count,
    xtol,
    rtol,
    maxiter,
):
    muB_values = np.full(temperatures.shape, np.nan)
    density_values = np.full(temperatures.shape, np.nan)
    previous_muB = float(muB_crit0)
    for index, temperature in enumerate(temperatures):
        if np.isclose(temperature, 0.0):
            root_muB = float(muB_crit0)
        else:
            root_muB = np.nan
            for half_width in bracket_half_widths:
                lower = max(float(muB_limits[0]), previous_muB - float(half_width))
                upper = min(float(muB_limits[1]), previous_muB + float(half_width))
                f_lower = _finite_pressure_residual(
                    lower, temperature, B14, ms, upB, nm_type
                )
                f_upper = _finite_pressure_residual(
                    upper, temperature, B14, ms, upB, nm_type
                )
                if np.isfinite(f_lower) and np.isfinite(f_upper) and f_lower * f_upper <= 0.0:
                    root_muB = float(
                        brentq(
                            lambda value: _finite_pressure_residual(
                                value, temperature, B14, ms, upB, nm_type
                            ),
                            lower,
                            upper,
                            xtol=float(xtol),
                            rtol=float(rtol),
                            maxiter=int(maxiter),
                        )
                    )
                    break
            if not np.isfinite(root_muB):
                probes = np.linspace(
                    float(muB_limits[0]),
                    float(muB_limits[1]),
                    int(fallback_probe_count),
                )
                residuals = np.asarray(
                    [
                        _finite_pressure_residual(
                            value, temperature, B14, ms, upB, nm_type
                        )
                        for value in probes
                    ],
                    dtype=float,
                )
                indices = np.flatnonzero(
                    np.isfinite(residuals[:-1])
                    & np.isfinite(residuals[1:])
                    & (residuals[:-1] * residuals[1:] <= 0.0)
                )
                if indices.size == 0:
                    raise RuntimeError(
                        f"No PNM-udsQM coexistence root at T={temperature:g} MeV"
                    )
                candidates = np.asarray(
                    [
                        brentq(
                            lambda value: _finite_pressure_residual(
                                value, temperature, B14, ms, upB, nm_type
                            ),
                            probes[bracket],
                            probes[bracket + 1],
                            xtol=float(xtol),
                            rtol=float(rtol),
                            maxiter=int(maxiter),
                        )
                        for bracket in indices
                    ],
                    dtype=float,
                )
                root_muB = float(candidates[np.argmin(np.abs(candidates - previous_muB))])
        muB_values[index] = root_muB
        density_values[index] = float(
            phase_velocity.nB_NM(
                root_muB,
                float(temperature),
                param=para.paraQMCRMF3,
                NM_type=str(nm_type),
            )
            / float(n0)
        )
        previous_muB = root_muB
    return density_values, muB_values


def _ud_transition_curve(
    temperatures,
    *,
    n0,
    B14,
    ms,
    upB,
    nm_type,
    initial_state,
    pressure_scale,
    density_scale,
    xtol,
    residual_tolerance,
):
    density_values = np.full(temperatures.shape, np.nan)
    muB_values = np.full(temperatures.shape, np.nan)
    muK_values = np.full(temperatures.shape, np.nan)

    def residual(state, temperature):
        muB_N, muK = np.asarray(state, dtype=float)
        muB_Q = muB_N - muK
        _, species = phase_velocity.nB_QM(
            muB_Q,
            muK,
            float(B14),
            float(temperature),
            ms=float(ms),
            upB=int(upB),
            return_species=True,
        )
        pressure = (
            phase_velocity.PNM(
                muB_N,
                float(temperature),
                param=para.paraQMCRMF3,
                NM_type=str(nm_type),
            )
            - phase_velocity.PQM(
                muB_Q,
                muK,
                float(B14),
                float(temperature),
                ms=float(ms),
                upB=int(upB),
            )
        ) / float(pressure_scale)
        return np.asarray((pressure, species["n_s"] / float(density_scale)))

    previous_state = np.asarray(initial_state, dtype=float)
    for index, temperature in enumerate(temperatures):
        solution = root(
            lambda state: residual(state, temperature),
            previous_state,
            method="hybr",
            options={"xtol": float(xtol)},
        )
        closing_residual = residual(solution.x, temperature)
        if (
            not solution.success
            or not np.all(np.isfinite(solution.x))
            or np.max(np.abs(closing_residual)) > float(residual_tolerance)
        ):
            raise RuntimeError(
                f"No PNM-udQM coexistence root at T={temperature:g} MeV: "
                f"{solution.message}; residual={closing_residual}"
            )
        muB_N, muK = np.asarray(solution.x, dtype=float)
        muB_values[index] = muB_N
        muK_values[index] = muK
        density_values[index] = float(
            phase_velocity.nB_NM(
                muB_N,
                float(temperature),
                param=para.paraQMCRMF3,
                NM_type=str(nm_type),
            )
            / float(n0)
        )
        previous_state = np.asarray(solution.x, dtype=float)
    return density_values, muB_values, muK_values


def load_or_calculate_overlay_curves(
    *,
    manifest_path,
    entropy_payload_path,
    cache_path,
    temperature_limits_MeV=(0.0, 115.0),
    uds_temperature_count=61,
    ud_temperature_count=21,
    force_recompute=False,
):
    """Return common coexistence curves plus the cached numerical-I2 entropy map."""
    manifest_path = Path(manifest_path)
    entropy_payload_path = Path(entropy_payload_path)
    cache_path = Path(cache_path)
    if cache_path.exists() and not force_recompute:
        return np.load(cache_path, allow_pickle=True).item()

    manifest = np.load(manifest_path, allow_pickle=True).item()
    entropy_payload = np.load(entropy_payload_path, allow_pickle=True).item()
    calibration = manifest["calibration"]
    n0 = float(calibration["n0_MeV3"])
    B14 = float(calibration["B14_MeV"])
    muB_crit0 = float(calibration["muB_crit0_MeV"])
    entropy_calibration = entropy_payload["calibration"]
    if not np.isclose(B14, float(entropy_calibration["B14_calibrated_MeV"]), rtol=1.0e-12):
        raise RuntimeError("Entropy map and boundary manifest use different bag constants")

    uds_temperature = np.linspace(
        float(temperature_limits_MeV[0]),
        float(temperature_limits_MeV[1]),
        int(uds_temperature_count),
    )
    ud_temperature = np.linspace(
        float(temperature_limits_MeV[0]),
        float(temperature_limits_MeV[1]),
        int(ud_temperature_count),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        uds_density, uds_muB = _uds_transition_curve(
            uds_temperature,
            n0=n0,
            B14=B14,
            muB_crit0=muB_crit0,
            ms=float(manifest["ms_MeV"]),
            upB=int(manifest["upB"]),
            nm_type=str(manifest["NM_type"]),
            muB_limits=(300.0, 1600.0),
            bracket_half_widths=(30.0, 60.0, 120.0, 240.0, 480.0),
            fallback_probe_count=65,
            xtol=1.0e-7,
            rtol=1.0e-10,
            maxiter=200,
        )
        ud_density, ud_muB, ud_muK = _ud_transition_curve(
            ud_temperature,
            n0=n0,
            B14=B14,
            ms=float(manifest["ms_MeV"]),
            upB=int(manifest["upB"]),
            nm_type=str(manifest["NM_type"]),
            initial_state=(1640.0, 587.0),
            pressure_scale=1.0e8,
            density_scale=1.0e6,
            xtol=1.0e-9,
            residual_tolerance=1.0e-7,
        )

    payload = {
        "uds_temperature_MeV": uds_temperature,
        "uds_nB_0minus_over_n0": uds_density,
        "uds_muB_MeV": uds_muB,
        "ud_temperature_MeV": ud_temperature,
        "ud_nB_0minus_over_n0": ud_density,
        "ud_muB_MeV": ud_muB,
        "ud_muK_MeV": ud_muK,
        "entropy_nB_0minus_over_n0": np.asarray(
            entropy_payload["nB_grid_over_n0"], dtype=float
        ),
        "entropy_temperature_MeV": np.asarray(
            entropy_payload["temperature_grid_MeV"], dtype=float
        ),
        "entropy_jump_per_baryon": np.asarray(
            entropy_payload["entropy_jump_grid"], dtype=float
        ),
        "entropy_jump_definition": str(entropy_payload["entropy_jump_definition"]),
        "B14_MeV": B14,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, payload, allow_pickle=True)
    return payload
