"""Rebuild the plot-only isothermal contour archive read by All_Plots_new.ipynb.

The notebook must stay free of cluster folders and of every EOS/BVP import, so
it reads pre-merged polylines from `isothermal-numerical-contour.npz`.  That
archive therefore has to be regenerated here whenever the merge logic in the
`26-09-16` / `26-09-21` workflows changes, or a newer checkpoint is copied in.

Run from the repository root:

    ./bin/python3 local_analysis_results/All_Plots_new/isothermal_contour_data/export_isothermal_contour_archive.py

Source `.npy` payloads are read from this directory, not from the workflow
folders, and are checked against the hashes recorded in `manifest.json`.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REPAIR_DIR = ROOT / "new_paper_calculations/26-09-16/isothermal_numeric_repair"
TARGETED_DIR = ROOT / "new_paper_calculations/26-09-21/isothermal_targeted"
COLD_NAME = "isothermal-cold-endpoints.npy"
COLD_SOURCE = ROOT / "new_paper_calculations/26-09-23/isothermal_cold_endpoints" / COLD_NAME
ARCHIVE_PATH = HERE / "isothermal-numerical-contour.npz"
MANIFEST_PATH = HERE / "manifest.json"
BOUNDARY_SAMPLES = 401
CURVE_SAMPLES = 361
NOTE = ("Numerical baseline plus accepted repair roots; only validated connected segments. "
        "No new BVP solves. The separate bridge_* arrays are interpolated across scheduled "
        "angles that hold no accepted root and must be drawn dashed. The curve_* arrays are "
        "one solid polyline per level with interior gaps and bracketed open ends filled, "
        "resampled with a monotone cubic through every scheduled angle; curve_solved_* "
        "marks which of those angles are accepted BVP roots. Every curve is anchored at "
        "the cold corner nB(0-) = 3 n0, T(0-) = 0, where the delta-muB = 0 boundary meets "
        "the temperature axis and the front speed vanishes.")

for path in (REPAIR_DIR, TARGETED_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from _repair_common import LEVELS  # noqa: E402
from plot_isothermal_contours import boundary_curve, continuous_level_curves  # noqa: E402
from plot_targeted_contours import accepted_root_groups, merged_targeted_segments, targeted_bridges  # noqa: E402


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_payload(name, kind):
    payload = np.load(HERE / name, allow_pickle=True).item()
    if payload.get("payload_kind") != kind:
        raise RuntimeError(f"{name} is not an {kind!r} payload")
    return payload


def main():
    manifest = json.loads(MANIFEST_PATH.read_text())
    baseline_sources = {name: record for name, record in manifest["source_files"].items()
                        if name != COLD_NAME}
    for name, record in baseline_sources.items():
        if sha256(HERE / name) != record["sha256"]:
            raise RuntimeError(f"{name} differs from the hash recorded in manifest.json")
    # The cold-endpoint checkpoint legitimately changes as more rays are solved,
    # so it is re-copied and re-hashed rather than checked against the manifest.
    # A run in progress still carries the repair payload kind; only the driver's
    # final stamp marks it complete, so a partial checkpoint is skipped here.
    if COLD_SOURCE.is_file() and np.load(
            COLD_SOURCE, allow_pickle=True).item().get("payload_kind") == "isothermal_cold_endpoints":
        shutil.copyfile(COLD_SOURCE, HERE / COLD_NAME)

    domain = load_payload("isothermal-domain.npy", "isothermal_domain")
    numerical = load_payload("isothermal-numerical.npy", "isothermal_numerical")
    repair = load_payload("isothermal-contour-repair.npy", "isothermal_contour_repair")
    cold = (load_payload(COLD_NAME, "isothermal_cold_endpoints")
            if (HERE / COLD_NAME).is_file() else None)
    if cold is not None and cold["domain_fingerprint"] != domain["domain_fingerprint"]:
        raise RuntimeError("The cold-endpoint checkpoint uses a different domain")
    targeted_path = HERE / "isothermal-targeted.npy"
    targeted = (load_payload("isothermal-targeted.npy", "isothermal_targeted")
                if targeted_path.exists() else None)
    fingerprints = {payload["domain_fingerprint"] for payload in
                    (domain, numerical, repair, *(() if targeted is None else (targeted,)))}
    if len(fingerprints) != 1:
        raise RuntimeError("Source payload domain fingerprints differ")

    velocity = np.asarray(numerical["velocity_m_s_grid"], float)
    status = np.asarray(numerical["task_status"], dtype=object)
    success = (status == "success") & np.isfinite(velocity) & (velocity > 0.)

    segments = merged_targeted_segments(domain, numerical, repair, targeted, levels=LEVELS)
    bridges = targeted_bridges(domain, repair, targeted, levels=LEVELS)
    roots = accepted_root_groups(repair, targeted)
    curves = continuous_level_curves(domain, repair, levels=LEVELS, samples=CURVE_SAMPLES,
                                     extra=() if cold is None else (cold,), cold_anchor=True)

    arrays = dict(
        levels_m_s=np.asarray(LEVELS, float),
        inner_boundary=np.column_stack(boundary_curve(domain, "delta_muB_equals_zero", BOUNDARY_SAMPLES)),
        outer_boundary=np.column_stack(boundary_curve(domain, "a_0plus_equals_one", BOUNDARY_SAMPLES)),
        baseline_density_grid=np.asarray(domain["nB_0minus_over_n0_grid"], float),
        baseline_temperature_grid=np.asarray(domain["T_0minus_grid_MeV"], float),
        baseline_velocity_grid=velocity,
        baseline_success_grid=success,
    )
    def pack(prefix, index, polylines):
        lines = [np.asarray(line, float).reshape(-1, 2) for line in polylines]
        arrays[f"{prefix}_{index}"] = np.concatenate(lines) if lines else np.zeros((0, 2), float)
        arrays[f"{prefix.replace('vertices', 'offsets')}_{index}"] = np.cumsum([0] + [len(line) for line in lines])

    for index, level in enumerate(LEVELS):
        pack("vertices", index, segments[index])
        pack("bridge_vertices", index, bridges[index])
        arrays[f"roots_{index}"] = np.asarray(roots.get(level) or [], float).reshape(-1, 2)
        # One solid polyline per level for the paper figure; the mask records
        # which of its nodes came from an accepted BVP root.
        vertices, solved = curves[index]
        arrays[f"curve_vertices_{index}"] = vertices
        arrays[f"curve_solved_{index}"] = solved

    metadata = dict(
        schema_version=manifest["schema_version"],
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        source_files={**baseline_sources, **({} if cold is None else {COLD_NAME: dict(
            source=str(COLD_SOURCE.relative_to(ROOT)), sha256=sha256(HERE / COLD_NAME))})},
        cold_endpoint_rays=0 if cold is None else sum(
            len(row.get("roots") or []) for row in cold["rays"].values()),
        baseline_successes=int(np.count_nonzero(success)),
        baseline_cells=int(success.size),
        repair_evaluations=len(repair["evaluations"]),
        repair_solutions=len(repair["points"]),
        accepted_contour_roots=sum(len(group) for group in roots.values()),
        interpolated_bridges=sum(len(level_bridges) for level_bridges in bridges),
        continuous_curve_nodes={f"{level:g}": [int(solved.sum()), int(solved.size)]
                                for level, (_, solved) in zip(LEVELS, curves)},
        targeted_present=targeted is not None,
        domain_fingerprint=domain["domain_fingerprint"],
        B_one_forth_MeV=float(domain["B_one_forth_MeV"]),
        ms_MeV=float(domain["ms_MeV"]),
        NM_type=str(domain["NM_type"]),
        note=NOTE,
    )
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(ARCHIVE_PATH, **arrays)

    MANIFEST_PATH.write_text(json.dumps(
        {**metadata, "plot_archive_sha256": sha256(ARCHIVE_PATH)}, indent=2) + "\n")
    for index, level in enumerate(LEVELS):
        lengths = np.diff(arrays[f"offsets_{index}"]).tolist()
        bridged = np.diff(arrays[f"bridge_offsets_{index}"]).tolist()
        solved = arrays[f"curve_solved_{index}"]
        print(f"  {level:>6g} m/s: {len(lengths)} solved polyline(s) {lengths}, "
              f"{len(bridged)} interpolated bridge(s) {bridged}, "
              f"{len(arrays[f'roots_{index}'])} accepted roots, "
              f"continuous curve {int(solved.sum())}/{solved.size} solved")
    print(f"Wrote {ARCHIVE_PATH.relative_to(ROOT)} and {MANIFEST_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
