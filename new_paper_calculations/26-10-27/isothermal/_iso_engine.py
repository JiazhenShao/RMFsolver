"""Point scheduler for the isothermal contour meshes (mesh-agnostic).

Each pending mesh point is evaluated in a persistent worker process; a worker
that exceeds the per-point timeout is killed and replaced, and the point is
recorded as 'timeout'. Only the parent writes: the payload is saved atomically
after every completed point, numerical profiles go to one small file per point.
Resume reuses every finished point and refuses a changed identity unless
restart=True, which writes a new suffixed payload instead.
"""
import importlib
import multiprocessing
from multiprocessing.connection import wait
from pathlib import Path
import time

import numpy as np

import _iso_common as common


def _resolve(evaluator):
    module, name = evaluator.split(":")
    return getattr(importlib.import_module(module), name)


def _worker(connection, evaluator):
    common._single_thread()
    function = _resolve(evaluator)
    while True:
        task = connection.recv()
        if task is None:
            return
        i, j, method, x, T = task
        started = time.perf_counter()
        try:
            record, profile = function(method, x, T)
        except Exception as exc:  # recorded, never raised in the scheduler
            record, profile = dict(status="unresolved", accepted=False, speed_m_s=np.nan,
                                   message=f"{type(exc).__name__}: {exc}"), None
        record["wall_s"] = time.perf_counter() - started
        connection.send((i, j, record, profile))


class _Slot:
    def __init__(self, context, evaluator):
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_worker, args=(child, evaluator), daemon=True)
        self.process.start()
        self.task, self.started = None, None

    def submit(self, task):
        self.task, self.started = task, time.perf_counter()
        self.connection.send(task)

    def kill(self):
        self.process.kill()
        self.process.join()


def _fresh_payload(mesh, identity, method):
    shape = mesh["x"].shape
    status = np.array(mesh["point_status"], dtype=object, copy=True)
    return dict(identity=identity, method=method, mesh=mesh, status=status,
                records=np.full(shape, None, dtype=object), speed_m_s=np.full(shape, np.nan),
                timings=dict(created=time.time(), wall_s=0.0))


def _restart_path(path):
    for k in range(1, 1000):
        candidate = path.with_name(f"{path.stem}-restart{k}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError("No free restart path")


def _seed(payload, seed_path, identity):
    """Copy finished points whose (x, T) match a compatible earlier payload exactly."""
    seed = common.load(seed_path)
    # Same solver bytes, physics, settings and method: accepted points are identical
    # results; workflow-module edits (engine, acceptance retries) do not change them.
    keys = ("physics", "xi", "numerical", "a_ceiling", "polar", "method", "evaluator")
    solver = "RMFsolver/phase_velocity.py"
    if (any(seed["identity"].get(k) != identity.get(k) for k in keys)
            or seed["identity"]["sources"][solver] != identity["sources"][solver]):
        raise RuntimeError(f"{seed_path} is not compatible apart from its mesh")
    key = lambda x, T: (round(float(x), 10), round(float(T), 8))
    finished = {key(seed["mesh"]["x"][i, j], seed["mesh"]["T"][i, j]): (i, j)
                for i, j in np.argwhere(np.isin(seed["status"].astype(str), ("accepted", "boundary_roundoff")))}
    source_profiles, target_profiles = Path(seed["profiles_dir"]), Path(payload["profiles_dir"])
    count = 0
    for i, j in np.argwhere(payload["status"] == "pending"):
        match = finished.get(key(payload["mesh"]["x"][i, j], payload["mesh"]["T"][i, j]))
        if match is None:
            continue
        payload["records"][i, j] = dict(seed["records"][match], seeded_from=str(seed_path))
        payload["status"][i, j] = seed["status"][match]
        payload["speed_m_s"][i, j] = seed["speed_m_s"][match]
        profile = source_profiles / f"{match[0]}_{match[1]}.npz"
        if profile.is_file():
            target_profiles.mkdir(parents=True, exist_ok=True)
            (target_profiles / f"{i}_{j}.npz").write_bytes(profile.read_bytes())
        count += 1
    return count


def run(mesh, out_path, method, workers, evaluator="_iso_evaluate:evaluate", timeout_s=120.0,
        restart=False, progress=False, seed_from=None):
    path = Path(out_path)
    identity = common.identity(method=method, mesh=mesh["fingerprint"], evaluator=evaluator)
    if path.is_file():
        payload = common.load(path)
        if payload["identity"] != identity:
            if not restart:
                raise RuntimeError(f"{path} was made with a different identity; use restart")
            path = _restart_path(path)
            payload = _fresh_payload(mesh, identity, method)
    else:
        payload = _fresh_payload(mesh, identity, method)
    payload.update(path=str(path), profiles_dir=str(path.with_name(path.stem + "-profiles")),
                   new_evaluations=0)
    payload["seeded"] = _seed(payload, seed_from, identity) if seed_from else 0
    return _schedule(payload, path, _tasks(payload, ("pending",)), evaluator, workers, timeout_s, progress)


def repair(out_path, workers, evaluator=None, timeout_s=120.0, statuses=("unresolved", "timeout"),
           progress=False):
    """Re-evaluate only non-accepted points in place with the current code.

    The payload identity is kept; each repaired record is marked and the pass is
    logged with the source hashes it ran with, so provenance stays explicit.
    """
    path = Path(out_path)
    payload = common.load(path)
    evaluator = evaluator or payload["identity"]["evaluator"]
    tasks = _tasks(payload, statuses)
    payload.update(path=str(path), profiles_dir=str(path.with_name(path.stem + "-profiles")),
                   new_evaluations=0)
    payload.setdefault("repairs", []).append(dict(time=time.time(), evaluator=evaluator,
                                                  statuses=statuses, sources=common.source_hashes(),
                                                  points=[(t[0], t[1]) for t in tasks]))
    return _schedule(payload, path, tasks, evaluator, workers, timeout_s, progress,
                     mark=lambda record: record.update(repaired=True))


def _tasks(payload, statuses):
    mesh, method = payload["mesh"], payload["method"]
    return [(int(i), int(j), method, float(mesh["x"][i, j]), float(mesh["T"][i, j]))
            for i, j in np.argwhere(np.isin(payload["status"].astype(str), statuses))]


def _schedule(payload, path, tasks, evaluator, workers, timeout_s, progress, mark=None):
    method, profiles_dir = payload["method"], Path(payload["profiles_dir"])
    total, started = len(tasks), time.perf_counter()
    context = multiprocessing.get_context("spawn")
    common._single_thread()
    slots = [_Slot(context, evaluator) for _ in range(min(workers, max(total, 1)))]
    try:
        while tasks or any(slot.task for slot in slots):
            for slot in slots:
                if slot.task is None and tasks:
                    slot.submit(tasks.pop(0))
            busy = {slot.connection: slot for slot in slots if slot.task}
            for connection in wait(list(busy), timeout=0.5):
                slot = busy[connection]
                i, j, record, profile = connection.recv()
                slot.task = None
                if mark:
                    mark(record)
                if profile is not None:
                    profiles_dir.mkdir(parents=True, exist_ok=True)
                    temporary = profiles_dir / f"{i}_{j}.tmp.npz"
                    np.savez(temporary, **profile)
                    temporary.replace(profiles_dir / f"{i}_{j}.npz")
                payload["records"][i, j] = record
                payload["status"][i, j] = record["status"]
                if record.get("accepted"):
                    payload["speed_m_s"][i, j] = record["speed_m_s"]
                payload["new_evaluations"] += 1
                payload["timings"]["wall_s"] += record.get("wall_s", 0.0)
                common.atomic_save(path, payload)
                if progress:
                    done = payload["new_evaluations"]
                    print(f"[{method}] {done}/{total} point ({i},{j}) x={record.get('x', np.nan):.4f} "
                          f"T={record.get('T', np.nan):.2f}: {record['status']} "
                          f"v={record.get('speed_m_s', np.nan):.5g} ({record.get('wall_s', 0):.1f}s, "
                          f"elapsed {time.perf_counter() - started:.0f}s)", flush=True)
            now = time.perf_counter()
            for index, slot in enumerate(slots):
                if slot.task and now - slot.started > timeout_s:
                    i, j = slot.task[:2]
                    slot.kill()
                    payload["records"][i, j] = dict(status="timeout", accepted=False, speed_m_s=np.nan,
                                                    message=f"exceeded {timeout_s} s")
                    payload["status"][i, j] = "timeout"
                    payload["new_evaluations"] += 1
                    common.atomic_save(path, payload)
                    slots[index] = _Slot(context, evaluator)
    finally:
        for slot in slots:
            if slot.process.is_alive():
                if slot.task is None:
                    slot.connection.send(None)
                    slot.process.join(timeout=5)
                if slot.process.is_alive():
                    slot.kill()
    common.atomic_save(path, payload)
    return payload
