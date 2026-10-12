#!/usr/bin/env python3
"""Export full 0minus/0plus EoS states on the T(0+)=5 MeV stable boundary."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

import RMFsolver.constants as const
from RMFsolver import phase_velocity


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_FILE = SCRIPT_DIR / "T0plus5-all-zero-upstream-v1-domain.npy"
OUTPUT_FILE = SCRIPT_DIR / "T0plus5-stable-neutron-boundary-EOS.csv"
TARGET_T_0PLUS_MEV = 5.0
BOUNDARY_DENSITY_RTOL = 2.0e-6
BOUNDARY_PRESSURE_RTOL = 2.0e-8
PNM_NK_DEFINITION = "pure_neutron_matter_nK_equals_nB"


def _records_for_temperature(records_by_temperature, target_temperature: float):
    for saved_temperature, records in records_by_temperature.items():
        if np.isclose(float(saved_temperature), target_temperature):
            return records
    raise RuntimeError(f"No stable-boundary records for T(0+)={target_temperature:g} MeV")


def _finite_float(value):
    value = float(value)
    if not np.isfinite(value):
        raise RuntimeError(f"Expected finite EoS value, received {value}")
    return value


def build_rows(payload: dict) -> list[dict]:
    calibration = payload["calibration"]
    B14 = float(calibration["B14_MeV"])
    n0 = float(calibration["n0_MeV3"])
    ms = float(payload["ms_MeV"])
    upB = int(payload["upB"])
    NM_type = str(payload["NM_type"])
    boundary = payload["boundary_diagnostics"]["semi_analytic"]
    records = _records_for_temperature(
        boundary["gap_curve_records_by_T0plus"], TARGET_T_0PLUS_MEV
    )

    rows = []
    for record in sorted(records, key=lambda item: float(item["theta_rad"])):
        T_0minus = float(record["T_MeV"])
        T_0plus = float(record["T_0plus_MeV"])
        muB_0minus = float(record["muB_0minus_MeV"])
        muB_0plus = float(record["muB_0plus_MeV"])
        muK_0plus = 0.0

        state_0minus = phase_velocity._analytic_nuclear_state(
            muB_0minus,
            T_0minus,
            NM_type=NM_type,
        )
        P_0minus = _finite_float(state_0minus["P_0minus"])
        epsilon_0minus = _finite_float(state_0minus["e_0minus"])
        h_0minus = _finite_float(state_0minus["h_0minus"])
        nB_0minus = _finite_float(state_0minus["nB_0minus"])
        nK_0minus = nB_0minus
        entropy_0minus = _finite_float(
            phase_velocity.sNM_n(nB_0minus, T_0minus, NM_type=NM_type)
        )

        quark_kinetic = phase_velocity._quark_uds_state(
            muB_0plus,
            muK_0plus,
            T_0plus,
            ms=ms,
            upB=upB,
        )
        P_0plus = _finite_float(
            phase_velocity.PQM(muB_0plus, muK_0plus, B14, T_0plus, ms=ms, upB=upB)
        )
        epsilon_0plus = _finite_float(
            phase_velocity.edensQM(
                muB_0plus, muK_0plus, B14, T_0plus, ms=ms, upB=upB
            )
        )
        h_0plus = P_0plus + epsilon_0plus
        entropy_0plus = _finite_float(
            phase_velocity.entropyQM(
                muB_0plus, muK_0plus, B14, T_0plus, ms=ms, upB=upB
            )
        )
        nB_0plus = _finite_float(quark_kinetic["nB"])
        nK_0plus = _finite_float(quark_kinetic["nK"])
        species = quark_kinetic["species"]
        chemical_potentials = quark_kinetic["chemical_potentials"]

        saved_nB_0minus = float(record["nB_over_n0"]) * n0
        saved_nB_0plus = float(record["nB_0plus_MeV3"])
        nB_0minus_relative_error = abs(nB_0minus - saved_nB_0minus) / saved_nB_0minus
        nB_0plus_relative_error = abs(nB_0plus - saved_nB_0plus) / saved_nB_0plus
        pressure_scale = max(abs(P_0minus), abs(P_0plus), 1.0)
        pressure_relative_residual = abs(P_0plus - P_0minus) / pressure_scale
        if nB_0minus_relative_error > BOUNDARY_DENSITY_RTOL:
            raise RuntimeError(
                f"0minus density reconstruction failed at theta={record['theta_deg']}: "
                f"relative error={nB_0minus_relative_error:.3e}"
            )
        if nB_0plus_relative_error > BOUNDARY_DENSITY_RTOL:
            raise RuntimeError(
                f"0plus density reconstruction failed at theta={record['theta_deg']}: "
                f"relative error={nB_0plus_relative_error:.3e}"
            )
        if pressure_relative_residual > BOUNDARY_PRESSURE_RTOL:
            raise RuntimeError(
                f"Isobar reconstruction failed at theta={record['theta_deg']}: "
                f"relative residual={pressure_relative_residual:.3e}"
            )

        row = {
            "theta_deg": float(record["theta_deg"]),
            "rho": float(record["rho"]),
            "boundary_kind": str(record["boundary_kind"]),
            "NM_type": NM_type,
            "B14_MeV": B14,
            "n0_MeV3": n0,
            "T_0minus_MeV": T_0minus,
            "muB_0minus_MeV": muB_0minus,
            "muK_0minus_MeV": "",
            "muK_0minus_defined": False,
            "nB_0minus_over_n0": nB_0minus / n0,
            "nB_0minus_MeV3": nB_0minus,
            "nB_0minus_fm-3": nB_0minus / const.hbarc3,
            "nK_0minus_MeV3": nK_0minus,
            "nK_0minus_fm-3": nK_0minus / const.hbarc3,
            "a_0minus": nK_0minus / nB_0minus,
            "nK_0minus_definition": PNM_NK_DEFINITION,
            "P_0minus_MeV4": P_0minus,
            "P_0minus_MeV_fm-3": P_0minus / const.hbarc3,
            "epsilon_0minus_MeV4": epsilon_0minus,
            "epsilon_0minus_MeV_fm-3": epsilon_0minus / const.hbarc3,
            "h_0minus_MeV4": h_0minus,
            "h_0minus_MeV_fm-3": h_0minus / const.hbarc3,
            "entropy_0minus_MeV3": entropy_0minus,
            "entropy_0minus_fm-3": entropy_0minus / const.hbarc3,
            "h_over_nB_0minus_MeV": h_0minus / nB_0minus,
            "entropy_over_nB_0minus": entropy_0minus / nB_0minus,
            "T_0plus_MeV": T_0plus,
            "muB_0plus_MeV": muB_0plus,
            "muK_0plus_MeV": muK_0plus,
            "mu_u_0plus_MeV": float(chemical_potentials["mu_u"]),
            "mu_d_0plus_MeV": float(chemical_potentials["mu_d"]),
            "mu_s_0plus_MeV": float(chemical_potentials["mu_s"]),
            "nB_0plus_over_n0": nB_0plus / n0,
            "nB_0plus_MeV3": nB_0plus,
            "nB_0plus_fm-3": nB_0plus / const.hbarc3,
            "nK_0plus_MeV3": nK_0plus,
            "nK_0plus_fm-3": nK_0plus / const.hbarc3,
            "a_0plus": nK_0plus / nB_0plus,
            "n_u_0plus_MeV3": float(species["n_u"]),
            "n_d_0plus_MeV3": float(species["n_d"]),
            "n_s_0plus_MeV3": float(species["n_s"]),
            "P_0plus_MeV4": P_0plus,
            "P_0plus_MeV_fm-3": P_0plus / const.hbarc3,
            "epsilon_0plus_MeV4": epsilon_0plus,
            "epsilon_0plus_MeV_fm-3": epsilon_0plus / const.hbarc3,
            "h_0plus_MeV4": h_0plus,
            "h_0plus_MeV_fm-3": h_0plus / const.hbarc3,
            "entropy_0plus_MeV3": entropy_0plus,
            "entropy_0plus_fm-3": entropy_0plus / const.hbarc3,
            "h_over_nB_0plus_MeV": h_0plus / nB_0plus,
            "entropy_over_nB_0plus": entropy_0plus / nB_0plus,
            "charge_neutrality_residual_MeV3": (
                2.0 * float(species["n_u"])
                - float(species["n_d"])
                - float(species["n_s"])
            ),
            "P_0plus_minus_P_0minus_MeV4": P_0plus - P_0minus,
            "h_over_nB_0plus_minus_0minus_MeV": (
                h_0plus / nB_0plus - h_0minus / nB_0minus
            ),
            "source_evaluation_count": int(record["evaluation_count"]),
        }
        rows.append(row)
    return rows


def main() -> int:
    payload = np.load(INPUT_FILE, allow_pickle=True).item()
    rows = build_rows(payload)
    if not rows:
        raise RuntimeError("Stable-boundary export produced no rows")
    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} stable-boundary EoS rows to {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
