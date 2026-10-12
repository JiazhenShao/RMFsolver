"""Portable, immutable warm-data snapshot for the log-temperature workflow.

Only historical accepted numerical solutions become point seeds. Plot curves
remain available in ``warm`` but are never interpreted as solved points.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from _logT_common import PHYSICS, atomic_save


SOURCE_NAMES = (
    "isothermal-domain.npy", "isothermal-numerical.npy",
    "isothermal-contour-repair.npy", "isothermal-cold-endpoints.npy",
    "isothermal-numerical-contour.npz",
)
PROBE_NAMES = (
    "core_mesh_T0.01_n3.001.npy", "core_mesh_T0.01_n3.01.npy",
    "core_mesh_T0.01_n3.05.npy",
)


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload(path, kind):
    payload = np.load(path, allow_pickle=True).item()
    if payload.get("payload_kind") != kind:
        raise ValueError(f"{path.name} has unexpected payload kind")
    return payload


def _copy_once(source, destination):
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite input snapshot: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=destination.name + ".", dir=destination.parent)
    os.close(fd)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _valid_profile(state):
    try:
        s = np.asarray(state["s"], dtype=float)
        nK = np.asarray(state["nK"], dtype=float)
        jK = np.asarray(state["jK"], dtype=float)
        return bool(s.ndim == 1 and s.size >= 2 and nK.shape == s.shape
                    and jK.shape == s.shape and s[0] == 0.0
                    and 0.0 < s[-1] < 1.0 and np.all(np.diff(s) > 0)
                    and np.all(np.isfinite(s)) and np.all(np.isfinite(nK))
                    and np.all(np.isfinite(jK)) and float(state["jB"]) > 0)
    except (KeyError, TypeError, ValueError):
        return False


def _accepted(result):
    return (isinstance(result, dict) and result.get("success") is True
            and result.get("bvp_status") == 0)


def _physics_matches(source, *, names):
    try:
        return all(
            (source[name] == expected if isinstance(expected, str)
             else np.isclose(float(source[name]), float(expected), rtol=1e-12, atol=0.0))
            for name, expected in names.items()
        )
    except (KeyError, TypeError, ValueError):
        return False


def _warm_physics(domain, numerical):
    fields = {"B_one_forth_MeV": PHYSICS["B_one_forth"],
              "ms_MeV": PHYSICS["ms"], "NM_type": PHYSICS["NM_type"],
              "upB": PHYSICS["upB"], "xi": PHYSICS["xi"]}
    if not all(_physics_matches(source, names=fields) for source in (domain, numerical)):
        raise ValueError("Warm source physics is incompatible with logT workflow")
    return dict(PHYSICS)


def _check_baseline_provenance(base, repair, cold):
    for source in (repair, cold):
        recorded = source.get("baseline_hashes", {})
        for name in SOURCE_NAMES[:2]:
            if recorded.get(name) != _hash(base / name):
                raise ValueError(f"Repair/cold baseline physics provenance mismatch: {name}")


def _probe_physics_matches(probe, result, repair):
    if not _physics_matches(probe, names={"B_one_forth": PHYSICS["B_one_forth"]}):
        return False
    explicit = probe.get("physics")
    if explicit is not None:
        return _physics_matches(explicit, names={
            "ms": PHYSICS["ms"], "NM_type": PHYSICS["NM_type"],
            "upB": PHYSICS["upB"], "xi": PHYSICS["xi"],
        })
    # The historical probes predate an explicit physics block. Accept only
    # their known solver identity plus physical closure evidence in the result.
    identity = probe.get("solver_identity", {}).get("sha256")
    repair_identity = repair.get("solver_identity", {}).get("sha256")
    return bool(identity and identity == repair_identity
                and result.get("a_0minus") == 1.0
                and result.get("a_0plus_source") == "maximum"
                and result.get("rate_model") == "exact_nonleptonic"
                and result.get("diffusion_model") == "local_muB_fixed_T"
                and result.get("nK_inf") == 0.0)


def _near_current_bound(row):
    result = row.get("result", {})
    return bool(row.get("current_bracket_near_bound", False)
                or row.get("near_current_bound", False)
                or (isinstance(result, dict) and result.get("current_bracket_near_bound", False)))


def _point(values, n0, *, profile_file=None):
    try:
        T = float(values["T_0minus"])
        density = float(values.get("nB_0minus_over_n0",
                                   float(values["nB_0minus"]) / n0))
        nB = float(values["nB_0minus"])
        speed = float(values["velocity_m_s"])
        jB = float(values["jB"])
    except (KeyError, TypeError, ValueError):
        return None
    if (not np.all(np.isfinite([T, density, nB, speed, jB]))
            or not (.01 <= T <= 20.0 and 1.0 <= density <= 5.5)
            or min(nB, speed, jB) <= 0
            or not np.isclose(nB, density * n0, rtol=1e-8)):
        return None
    row = dict(id=f"{T.hex()}:{density.hex()}", T_0minus=T,
               nB_0minus_over_n0=density, nB_0minus=nB,
               velocity_m_s=speed, jB=jB)
    if profile_file is not None:
        row["profile_file"] = profile_file
    return row


def _load_verified(directory):
    directory = Path(directory)
    manifest_path = directory / "inputs/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != 1:
        raise ValueError("Unsupported input manifest schema")
    if manifest.get("physics") != PHYSICS:
        raise ValueError("Input manifest physics is incompatible with logT workflow")
    for relative, expected in manifest["files"].items():
        path = directory / relative
        if not path.is_file() or _hash(path) != expected["sha256"]:
            raise ValueError(f"Input hash mismatch: {relative}")
    return manifest


def load_inputs(directory):
    """Verify the snapshot and return warm arrays plus physical numerical seeds."""
    directory = Path(directory)
    manifest = _load_verified(directory)
    base = directory / "inputs"
    domain = _payload(base / SOURCE_NAMES[0], "isothermal_domain")
    numerical = _payload(base / SOURCE_NAMES[1], "isothermal_numerical")
    repair = _payload(base / SOURCE_NAMES[2], "isothermal_contour_repair")
    cold = _payload(base / SOURCE_NAMES[3], "isothermal_cold_endpoints")
    _warm_physics(domain, numerical)
    _check_baseline_provenance(base, repair, cold)
    fingerprint = domain["domain_fingerprint"]
    if fingerprint != manifest["fingerprint"] or any(
        source["domain_fingerprint"] != fingerprint
        for source in (numerical, repair, cold)
    ):
        raise ValueError("Warm input domain fingerprints differ")
    ratios = np.asarray(domain["nB_0minus_over_n0_grid"], float)
    densities = np.asarray(domain["nB_0minus_grid"], float)
    n0_candidates = densities[ratios > 0] / ratios[ratios > 0]
    n0 = float(np.median(n0_candidates))
    if not np.isfinite(n0) or n0 <= 0 or not np.allclose(n0_candidates, n0, rtol=1e-8):
        raise ValueError("Domain has inconsistent saturation density")
    if not np.isclose(n0, float(manifest["n0"]), rtol=1e-12):
        raise ValueError("Input manifest saturation density changed")
    points = {}
    status = np.asarray(numerical["task_status"])
    records = np.asarray(numerical["records"], dtype=object)
    if status.shape != records.shape:
        raise ValueError("Warm numerical status/record shape mismatch")
    for index in np.ndindex(status.shape):
        record = records[index]
        if status[index] != "success" or not _accepted(record) or _near_current_bound(record):
            continue
        row = _point(record, n0)
        if row is not None:
            row["source"] = f"snapshot:inputs/{SOURCE_NAMES[1]}"
            points[row["id"]] = row
    for source, name in ((repair, SOURCE_NAMES[2]), (cold, SOURCE_NAMES[3])):
        for historical in source["points"].values():
            if not _accepted(historical.get("result")) or _near_current_bound(historical):
                continue
            row = _point(historical, n0)
            if row is not None:
                row["source"] = f"snapshot:inputs/{name}"
                points[row["id"]] = row
    for relative in manifest.get("profiles", []):
        profile = np.load(directory / relative, allow_pickle=True).item()
        point_data = manifest["files"][relative].get("point", profile.get("point"))
        row = _point(point_data, n0, profile_file=relative)
        if row is not None:
            row["source"] = f"snapshot:{relative}"
            points[row["id"]] = row
    with np.load(base / SOURCE_NAMES[4], allow_pickle=False) as archive:
        warm = {key: archive[key].copy() for key in archive.files}
    return dict(warm=warm, points=list(points.values()), n0=n0,
                fingerprint=fingerprint, manifest=manifest)


def bundle_inputs(project_root, directory):
    """Snapshot historical sources once; subsequent calls only verify them."""
    project_root, directory = Path(project_root), Path(directory)
    manifest_path = directory / "inputs/manifest.json"
    if manifest_path.exists():
        return load_inputs(directory)
    base = directory / "inputs"
    if base.exists() and any(base.iterdir()):
        raise FileExistsError(f"Incomplete input snapshot already exists: {base}")
    source_dir = project_root / "local_analysis_results/All_Plots_new/isothermal_contour_data"
    for name in SOURCE_NAMES:
        _copy_once(source_dir / name, base / name)
    domain = _payload(base / SOURCE_NAMES[0], "isothermal_domain")
    numerical = _payload(base / SOURCE_NAMES[1], "isothermal_numerical")
    physics = _warm_physics(domain, numerical)
    repair = _payload(base / SOURCE_NAMES[2], "isothermal_contour_repair")
    cold = _payload(base / SOURCE_NAMES[3], "isothermal_cold_endpoints")
    _check_baseline_provenance(base, repair, cold)
    ratios = np.asarray(domain["nB_0minus_over_n0_grid"], float)
    densities = np.asarray(domain["nB_0minus_grid"], float)
    n0 = float(np.median(densities[ratios > 0] / ratios[ratios > 0]))
    files = {f"inputs/{name}": dict(source=str((source_dir / name).relative_to(project_root)),
                                      sha256=_hash(base / name)) for name in SOURCE_NAMES}
    profiles = []
    cache_directories = (
        (repair, "isothermal-contour-repair-profiles",
         project_root / "new_paper_calculations/26-09-16/isothermal_numeric_repair"),
        (cold, "isothermal-cold-endpoints-profiles",
         project_root / "new_paper_calculations/26-09-23/isothermal_cold_endpoints"),
    )
    for source_payload, expected_folder, cache_root in cache_directories:
        for historical in source_payload["points"].values():
            if not _accepted(historical.get("result")) or _near_current_bound(historical):
                continue
            point_data = _point(historical, n0)
            if point_data is None:
                continue
            named = historical.get("profile_file")
            if not isinstance(named, str):
                continue
            named_path = Path(named)
            if (len(named_path.parts) != 2 or named_path.parts[0] != expected_folder
                    or named_path.suffix != ".npy"):
                continue
            source = cache_root / named_path
            if not source.is_file():
                continue
            try:
                profile = np.load(source, allow_pickle=True).item()
                if not _valid_profile(profile) or not all(
                    np.isclose(float(profile[key]), float(point_data[key]), rtol=1e-8, atol=0.0)
                    for key in ("T_0minus", "nB_0minus", "jB")
                ):
                    continue
            except (KeyError, TypeError, ValueError):
                continue
            relative = f"inputs/profiles/{expected_folder}-{named_path.name}"
            if relative in files:
                continue
            _copy_once(source, directory / relative)
            profiles.append(relative)
            files[relative] = dict(source=str(source.relative_to(project_root)),
                                   source_sha256=_hash(source),
                                   sha256=_hash(directory / relative), point=point_data)
    probe_dir = project_root / "local_analysis_results/isothermal_low_temperature_probe"
    for name in PROBE_NAMES:
        source = probe_dir / name
        if not source.is_file():
            continue
        probe = np.load(source, allow_pickle=True).item()
        result = probe.get("result", {})
        state = result.get("continuation_state")
        residual = float(result.get("endpoint_flux_residual_max", np.inf))
        if not (_accepted(result) and _valid_profile(state)
                and _probe_physics_matches(probe, result, repair)
                and np.isfinite(residual) and residual <= 1e-8):
            continue
        point_data = dict(T_0minus=probe["T_0minus"],
                          nB_0minus_over_n0=probe["nB_0minus_over_n0"],
                          nB_0minus=result["nB_0minus"],
                          velocity_m_s=result["velocity_m_s"], jB=result["jB"])
        if _point(point_data, n0) is None:
            continue
        relative = f"inputs/profiles/{name}"
        payload = dict(point=point_data, **state)
        atomic_save(directory / relative, payload)
        profiles.append(relative)
        files[relative] = dict(source=str(source.relative_to(project_root)),
                               source_sha256=_hash(source),
                               sha256=_hash(directory / relative), point=point_data)
    manifest = dict(schema=1, fingerprint=domain["domain_fingerprint"], n0=n0,
                    physics=physics,
                    files=files, profiles=profiles)
    fd, temporary = tempfile.mkstemp(prefix="manifest.json.", dir=base)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, manifest_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return load_inputs(directory)


def _project_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "RMFsolver").is_dir():
            return parent
    raise FileNotFoundError("Could not locate project root containing RMFsolver")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-project", action="store_true")
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    if not args.bundle_project:
        parser.error("Pass --bundle-project to create or verify a snapshot")
    loaded = bundle_inputs(_project_root(), args.directory)
    print(f"{len(loaded['points'])} cached cold numerical seed points; "
          f"fingerprint {loaded['fingerprint']}")
