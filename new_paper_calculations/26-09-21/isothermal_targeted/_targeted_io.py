"""Immutable input snapshots and portable preflight for designated shots."""
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from _targeted import OLD, source_roots, initial_targets
from _repair_common import atomic_save, load, load_baseline, sha256

REPAIR_NAME = "isothermal-contour-repair.npy"


def read_inputs(source, *, check_live=True):
    source=Path(source)
    baseline,hashes=load_baseline(source,check_live=check_live)
    repair=load(source/REPAIR_NAME)
    if repair.get("payload_kind")!="isothermal_contour_repair" or repair.get("baseline_hashes")!=hashes:
        raise ValueError("Repair checkpoint is incompatible with its baseline")
    if repair["domain_fingerprint"]!=baseline[0]["domain_fingerprint"]:
        raise ValueError("Repair domain fingerprint differs")
    roots=source_roots(repair)
    initial_targets(roots)
    if check_live:
        from RMFsolver import phase_velocity as pv
        if sha256(pv.__file__)!=repair["solver_identity"]["sha256"]:
            raise ValueError("Live solver differs from the saved repair version; do not silently reuse its profiles")
    hashes[REPAIR_NAME]=sha256(source/REPAIR_NAME)
    return baseline,repair,roots,hashes


def snapshot_inputs(directory,source):
    """Only calculation mode snapshots. Original source and profiles are untouched."""
    directory,source=Path(directory),Path(source)
    destination=directory/"inputs"
    if destination.exists():
        _,_,_,hashes=read_inputs(destination)
        if load(destination/"manifest.npy")["hashes"]!=hashes:
            raise ValueError("Immutable targeted input snapshot changed")
        return destination
    _,_,_,hashes=read_inputs(source)
    directory.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="input-snapshot-",dir=directory) as temporary:
        for name,digest in hashes.items():
            target=Path(temporary)/name
            shutil.copyfile(source/name,target)
            if sha256(target)!=digest:
                raise ValueError("Source checkpoint changed while being copied; retry snapshot")
        # Verify cross-file relationships before publishing the directory.
        read_inputs(temporary)
        atomic_save(Path(temporary)/"manifest.npy",dict(hashes=hashes))
        os.rename(temporary,destination)
    return destination


def identity(directory,domain,repair,hashes):
    directory=Path(directory)
    return dict(input_hashes=hashes,domain_fingerprint=domain["domain_fingerprint"],
                solver_identity=repair["solver_identity"],
                workflow_hashes={name:sha256(directory/name) for name in ("_targeted.py","_targeted_io.py","_targeted_trial.py")},
                numerical_trial_sha256=sha256(OLD/"_repair_common.py"))


def profile_count(repair,cache):
    cache=Path(cache)
    return sum((cache/Path(p["profile_file"]).name).is_file() for p in repair["points"].values() if p.get("profile_file"))


def verify_snapshot(directory):
    directory=Path(directory)
    expected=load(directory/"manifest.npy")["hashes"]
    if any(sha256(directory/name)!=digest for name,digest in expected.items()):
        raise ValueError("An immutable input changed during targeted work")
