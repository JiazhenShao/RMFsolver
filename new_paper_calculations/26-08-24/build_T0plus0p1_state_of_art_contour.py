#!/usr/bin/env python3
"""Consolidate the state-of-the-art T(0+)=0.1 MeV numerical contour."""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
CALCULATIONS_ROOT = SCRIPT_DIR.parent
DOMAIN_PATH = CALCULATIONS_ROOT / "26-08-21" / "T0plus0p1-domain.npy"
CORE_RESULT_PATH = (
    CALCULATIONS_ROOT / "26-08-21" / "T0plus0p1-uNmax-angular-wavefront.npy"
)
RHO086_RESULT_PATH = (
    SCRIPT_DIR
    / "rho0p86_full_theta_shell"
    / "T0plus0p1-rho0p86-full-theta-shell.npy"
)
ADAPTIVE_RESULT_PATH = (
    SCRIPT_DIR
    / "rho5k_adaptive_bracket"
    / "T0plus0p1-rho5k-adaptive-bracket.npy"
)
OUTPUT_PATH = SCRIPT_DIR / "T0plus0p1-uNmax-state-of-art-complete-5km.npy"
T0PLUS_MEV = 0.1
CORE_MAX_RHO = 0.84
RHO086 = 0.86
TARGET_CONTOUR_M_S = 5000.0
SPEED_OF_LIGHT_M_S = 299_792_458.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_payload(path: Path) -> dict[str, Any]:
    return np.load(path, allow_pickle=True).item()


def tagged_record(
    record: dict[str, Any], *, source_dataset: str, source_path: Path
) -> dict[str, Any]:
    result = dict(record)
    result["source_dataset"] = str(source_dataset)
    result["source_path"] = str(source_path)
    result["u_0minus_m_s"] = (
        float(result["u_0minus"]) * SPEED_OF_LIGHT_M_S
    )
    return result


def point_key(record: dict[str, Any]) -> tuple[int, float]:
    return (
        int(record["theta_index"]),
        round(float(record["radial_fraction_used"]), 10),
    )


def build_payload() -> dict[str, Any]:
    domain = load_payload(DOMAIN_PATH)
    core = load_payload(CORE_RESULT_PATH)
    rho086 = load_payload(RHO086_RESULT_PATH)
    adaptive = load_payload(ADAPTIVE_RESULT_PATH)

    if not bool(rho086.get("run_complete")):
        raise RuntimeError("rho=0.86 source checkpoint is incomplete")
    if not bool(adaptive.get("run_complete")):
        raise RuntimeError("adaptive 5000 m/s source checkpoint is incomplete")
    if not np.all(np.asarray(adaptive["accepted_5000_bracket"], dtype=bool)):
        raise RuntimeError("adaptive checkpoint does not bracket 5000 m/s at every target")

    core_records = [
        tagged_record(
            record,
            source_dataset="26-08-21_complete_core_through_rho0p84",
            source_path=CORE_RESULT_PATH,
        )
        for record in core["radial_shell_records"]
        if bool(record.get("success"))
        and float(record["radial_fraction_used"]) <= CORE_MAX_RHO + 1.0e-12
    ]
    rho086_records = [
        tagged_record(
            record,
            source_dataset="26-08-24_complete_rho0p86_shell",
            source_path=RHO086_RESULT_PATH,
        )
        for record in rho086["records"]
        if bool(record.get("success"))
    ]
    adaptive_records = [
        tagged_record(
            record,
            source_dataset="26-08-24_adaptive_5km_hidden_points",
            source_path=ADAPTIVE_RESULT_PATH,
        )
        for record in adaptive["records"]
        if bool(record.get("success"))
    ]
    all_records = core_records + rho086_records + adaptive_records
    keys = [point_key(record) for record in all_records]
    if len(keys) != len(set(keys)):
        duplicates = sorted(key for key in set(keys) if keys.count(key) > 1)
        raise RuntimeError(f"duplicate numerical points found: {duplicates[:8]}")
    all_records.sort(
        key=lambda record: (
            int(record["theta_index"]),
            float(record["radial_fraction_used"]),
        )
    )

    radial_axis = np.asarray(core["contour_radial_fraction_axis"], dtype=float)
    core_columns = radial_axis <= CORE_MAX_RHO + 1.0e-12
    core_rho_axis = radial_axis[core_columns]
    density_grid = np.asarray(core["nB_0minus_over_n0_grid"], dtype=float)[
        :, core_columns, 0
    ]
    temperature_grid = np.asarray(core["T_0minus_MeV_grid"], dtype=float)[
        :, core_columns, 0
    ]
    velocity_grid = np.asarray(
        core["numerical_outward_u_0minus_m_s_grid"], dtype=float
    )[:, core_columns, 0]
    jB_grid = np.asarray(core["numerical_outward_jB_grid"], dtype=float)[
        :, core_columns, 0
    ]
    success_grid = np.asarray(
        core["numerical_outward_success_grid"], dtype=bool
    )[:, core_columns, 0]
    if density_grid.shape != (30, 17) or not np.all(success_grid):
        raise RuntimeError("expected a complete 30 x 17 numerical core")

    rho086_by_theta = {
        int(record["theta_index"]): record for record in rho086_records
    }
    if tuple(sorted(rho086_by_theta)) != tuple(range(30)):
        raise RuntimeError("rho=0.86 shell does not cover all 30 theta indices")
    outer_density = np.asarray(
        [rho086_by_theta[index]["nB_0minus_over_n0"] for index in range(30)],
        dtype=float,
    )
    outer_temperature = np.asarray(
        [rho086_by_theta[index]["T_0minus_MeV"] for index in range(30)],
        dtype=float,
    )
    outer_velocity = np.asarray(
        [rho086_by_theta[index]["u_0minus_m_s"] for index in range(30)],
        dtype=float,
    )
    outer_jB = np.asarray(
        [rho086_by_theta[index]["jB"] for index in range(30)], dtype=float
    )
    outer_rho = np.full(30, RHO086, dtype=float)
    outer_source = np.full(30, "rho0p86_shell", dtype="U32")

    adaptive_final = {
        int(record["theta_index"]): tagged_record(
            record,
            source_dataset="26-08-24_adaptive_5km_terminal",
            source_path=ADAPTIVE_RESULT_PATH,
        )
        for record in adaptive["final_records"]
        if bool(record.get("success"))
    }
    adaptive_theta = tuple(
        int(value) for value in np.asarray(adaptive["theta_indices"], dtype=int)
    )
    if tuple(sorted(adaptive_final)) != adaptive_theta:
        raise RuntimeError("adaptive terminal records do not cover the target theta range")
    for theta_index in adaptive_theta:
        record = adaptive_final[theta_index]
        outer_density[theta_index] = float(record["nB_0minus_over_n0"])
        outer_temperature[theta_index] = float(record["T_0minus_MeV"])
        outer_velocity[theta_index] = float(record["u_0minus_m_s"])
        outer_jB[theta_index] = float(record["jB"])
        outer_rho[theta_index] = float(record["radial_fraction_used"])
        outer_source[theta_index] = "adaptive_5km_terminal"
    if not np.all(outer_velocity > TARGET_CONTOUR_M_S):
        failures = np.flatnonzero(outer_velocity <= TARGET_CONTOUR_M_S)
        raise RuntimeError(
            f"outer numerical shell fails to bracket 5000 m/s at theta {failures.tolist()}"
        )

    density_grid = np.concatenate((density_grid, outer_density[:, None]), axis=1)
    temperature_grid = np.concatenate(
        (temperature_grid, outer_temperature[:, None]), axis=1
    )
    velocity_grid = np.concatenate((velocity_grid, outer_velocity[:, None]), axis=1)
    jB_grid = np.concatenate((jB_grid, outer_jB[:, None]), axis=1)
    success_grid = np.concatenate(
        (success_grid, np.ones((30, 1), dtype=bool)), axis=1
    )
    rho_grid = np.concatenate(
        (
            np.broadcast_to(core_rho_axis[None, :], (30, core_rho_axis.size)),
            outer_rho[:, None],
        ),
        axis=1,
    )
    source_grid = np.full(density_grid.shape, "26-08-21_core", dtype="U32")
    source_grid[:, -1] = outer_source
    theta_rad_axis = np.asarray(
        [
            point["theta_rad"]
            for point in sorted(
                (
                    point
                    for point in domain["points"]
                    if int(point["radial_index"]) == 0
                ),
                key=lambda point: int(point["theta_index"]),
            )
        ],
        dtype=float,
    )
    theta_index_grid = np.broadcast_to(
        np.arange(30, dtype=int)[:, None], density_grid.shape
    ).copy()
    theta_rad_grid = np.broadcast_to(
        theta_rad_axis[:, None], density_grid.shape
    ).copy()

    all_theta = np.asarray(
        [record["theta_index"] for record in all_records], dtype=int
    )
    all_rho = np.asarray(
        [record["radial_fraction_used"] for record in all_records], dtype=float
    )
    all_density = np.asarray(
        [record["nB_0minus_over_n0"] for record in all_records], dtype=float
    )
    all_temperature = np.asarray(
        [record["T_0minus_MeV"] for record in all_records], dtype=float
    )
    all_velocity = np.asarray(
        [record["u_0minus_m_s"] for record in all_records], dtype=float
    )
    all_jB = np.asarray([record["jB"] for record in all_records], dtype=float)

    return {
        "run_tag": "T0plus0p1_uNmax_state_of_art_complete_5km",
        "created_at_utc": utc_now(),
        "T_0plus_MeV": T0PLUS_MEV,
        "solver_method": "full_numerical_uNmax_energy_conserving_nK",
        "composition_definition": "nK_over_local_nB",
        "current_definition": "u_nK_minus_D_dnK_dx",
        "source_paths": {
            "domain": str(DOMAIN_PATH),
            "core": str(CORE_RESULT_PATH),
            "rho0p86_shell": str(RHO086_RESULT_PATH),
            "adaptive_5km": str(ADAPTIVE_RESULT_PATH),
        },
        "profiles_included": False,
        "profile_source_paths": (
            str(CORE_RESULT_PATH),
            str(RHO086_RESULT_PATH),
            str(ADAPTIVE_RESULT_PATH),
        ),
        "all_successful_records": all_records,
        "all_successful_point_count": int(len(all_records)),
        "all_successful_theta_index": all_theta,
        "all_successful_rho": all_rho,
        "all_successful_nB_0minus_over_n0": all_density,
        "all_successful_T_0minus_MeV": all_temperature,
        "all_successful_u_0minus_m_s": all_velocity,
        "all_successful_jB": all_jB,
        "plot_grid_definition": (
            "complete numerical core through rho=0.84 plus one fully numerical "
            "outer shell bracketing 5000 m/s at every theta"
        ),
        "plot_grid_shape": tuple(int(value) for value in density_grid.shape),
        "plot_theta_rad_axis": theta_rad_axis,
        "plot_theta_index_grid": theta_index_grid,
        "plot_theta_rad_grid": theta_rad_grid,
        "plot_rho_grid": rho_grid,
        "plot_point_source_grid": source_grid,
        "plot_nB_0minus_over_n0_grid": density_grid,
        "plot_T_0minus_MeV_grid": temperature_grid,
        "plot_u_0minus_m_s_grid": velocity_grid,
        "plot_jB_grid": jB_grid,
        "plot_success_grid": success_grid,
        "outer_bracket_theta_indices": np.arange(30, dtype=int),
        "outer_bracket_rho": outer_rho,
        "outer_bracket_nB_0minus_over_n0": outer_density,
        "outer_bracket_T_0minus_MeV": outer_temperature,
        "outer_bracket_u_0minus_m_s": outer_velocity,
        "outer_bracket_jB": outer_jB,
        "outer_bracket_source": outer_source,
        "complete_5000_contour_bracket": True,
        "minimum_outer_bracket_velocity_m_s": float(np.min(outer_velocity)),
        "boundary_diagnostics": domain["boundary_diagnostics"],
        "transform": domain["transform"],
        "calibration": domain["calibration"],
    }


def atomic_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".npy", delete=False
    ) as handle:
        temporary = Path(handle.name)
        np.save(handle, payload, allow_pickle=True)
    temporary.replace(path)


def main() -> int:
    payload = build_payload()
    atomic_save(OUTPUT_PATH, payload)
    print(f"Saved {OUTPUT_PATH}")
    print(f"Successful numerical points: {payload['all_successful_point_count']}")
    print(f"Ready-to-plot grid: {payload['plot_grid_shape']}")
    print(
        "Minimum outer bracket velocity: "
        f"{payload['minimum_outer_bracket_velocity_m_s']:.2f} m/s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
