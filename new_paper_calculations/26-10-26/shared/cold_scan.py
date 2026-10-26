"""Resumable exact-cold contour tracing, shared by local and cluster runners."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import signal
import tempfile
import threading
import time

# Set before importing NumPy or spawning children; one BLAS thread per call.
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import numpy as np

from bounded_call import run_module_call_with_hard_timeout
from cold_physics import METHODS

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEFAULT_INPUT = HERE.parent / 'inputs/cold-scan-seeds.json'
LEVELS = (50., 100., 200., 500., 1000., 2000., 5000.)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npy', delete=False) as f:
        temp = Path(f.name)
        np.save(f, payload, allow_pickle=True)
    temp.replace(path)


def cluster_workers():
    for name in ('SLURM_CPUS_PER_TASK', 'PBS_NP', 'NSLOTS'):
        try:
            n = int(os.environ[name])
            if n > 0:
                return n
        except (KeyError, ValueError):
            pass
    return len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)


def point_on_ray(theta, rho):
    return dict(theta_rad=float(theta), rho=float(rho),
        nB_0minus_over_n0=float(1 + 4.5*rho*np.cos(theta)),
        T_0minus_MeV=0. if theta == 0 else float(120*rho*np.sin(theta)))


def ray_limit(theta):
    return min(1 / max(np.cos(theta), 1e-16), 115 / (120*max(np.sin(theta), 1e-16)))


def coordinate_key(method, theta, rho):
    return method + '-' + hashlib.sha256(f'{theta:.14f}:{rho:.14f}'.encode()).hexdigest()[:24]


def normalized_xy(point):
    return np.array([(point['nB_0minus_over_n0']-1)/4.5, point['T_0minus_MeV']/115])


def compatible(existing, identity):
    if existing.get('identity') != identity:
        raise ValueError('Checkpoint settings, temperature, inputs or source hashes differ; use a new --output')


class Scan:
    def __init__(self, output, identity, seeds, timeout):
        self.output = Path(output)
        self.record_dir = self.output.with_name(self.output.stem + '-records')
        self.identity, self.seeds, self.timeout = identity, seeds, timeout
        self.cancel = threading.Event()
        self.lock = threading.Lock()
        self.cache = {}
        self.calls = 0
        self.started = time.monotonic()

    def evaluate(self, method, theta, rho, *, geometry_only=False):
        if self.cancel.is_set():
            raise InterruptedError('Calculation cancelled')
        key = coordinate_key(method, theta, rho)
        if geometry_only:
            key = 'boundary-' + key
        if key in self.cache:
            return self.cache[key]
        file = self.record_dir / (key + '.npy')
        previous = None
        if file.exists():
            previous = np.load(file, allow_pickle=True).item()
            compatible(previous, self.identity)
            if previous['status'] != 'failed':
                self.cache[key] = previous
                return previous
        point = point_on_ray(theta, rho)
        try:
            row = run_module_call_with_hard_timeout('cold_physics', 'classify' if geometry_only else 'evaluate',
                args=(method, point, self.identity['physics']), timeout_s=self.timeout,
                cancel=self.cancel)
        except InterruptedError:
            raise
        except Exception as exc:
            row = dict(method=method, point=point, status='failed', speed=np.nan,
                       message=str(exc), attempts=[])
        row.update(identity=self.identity, record_key=key)
        if previous is not None:
            row['previous_failures'] = previous.get('previous_failures', []) + [
                {k: previous.get(k) for k in ('message', 'result', 'attempts')}]
        atomic_save(file, row)  # Every terminal call survives an interrupted ray.
        self.cache[key] = row
        with self.lock:
            self.calls += 1
            if self.calls % 50 == 0:
                print(f'{self.calls} new method calls; {time.monotonic()-self.started:.0f} s', flush=True)
        return row

    def geometry(self, method, theta, rho):
        return self.evaluate(method, theta, rho, geometry_only=True)

    def trace_ray(self, method, theta):
        if self.cancel.is_set():
            raise InterruptedError('Calculation cancelled')
        limit = ray_limit(theta)
        samples = list(np.linspace(0, limit, 17))
        for p in self.seeds:
            if abs(p['theta_rad']-theta) < 1e-12:
                rho = np.hypot((p['nB_0minus_over_n0']-1)/4.5, p['T_0minus_MeV']/120)
                if 0 < rho < limit:
                    samples.append(float(rho))
        rows = {}

        def call(rho):
            rho = float(rho)
            if rho not in rows:
                rows[rho] = self.evaluate(method, theta, rho)
            return rows[rho]

        for rho in sorted(set(samples)):
            call(rho)
        boundaries = {}
        boundary_records = []
        boundary_failures = []
        # Refine physical transitions; failed evaluations never become boundaries.
        initial = sorted(rows)
        for lo, hi in zip(initial[:-1], initial[1:]):
            left, right = rows[lo], rows[hi]
            if {left['status'], right['status']} != {'accepted', 'excluded'}:
                continue
            excluded = left if left['status'] == 'excluded' else right
            name = excluded['exclusion']
            left_status = 'interior' if left['status']=='accepted' else 'excluded'
            for _ in range(18):
                if hi-lo <= 1e-4:
                    break
                mid = (lo+hi)/2
                middle = self.geometry(method,theta,mid)
                boundary_records.append(middle['record_key'])
                if middle['status'] == 'failed':
                    boundary_failures.append(middle['record_key'])
                    break
                if middle['status'] == left_status:
                    lo = mid
                else:
                    hi = mid
            boundaries[name] = dict(point=point_on_ray(theta, (lo+hi)/2),
                rho_bracket=(lo, hi), resolved=hi-lo <= 1e-4)

        # Only request speeds far enough inside to bracket the plotted levels.
        # Close-to-zero I2 tables are unnecessary for locating the phase boundary.
        for name, target, inward in (('stable_matter',min(LEVELS),True),
                                     ('composition_ceiling',max(LEVELS),False)):
            boundary = boundaries.get(name)
            if not boundary or not boundary['resolved']:
                continue
            for _ in range(20):
                accepted = sorted((r for r in rows.values() if r['status']=='accepted'),
                                  key=lambda r:r['point']['rho'])
                if not accepted:
                    break
                anchor = accepted[0] if inward else accepted[-1]
                if (anchor['speed'] <= target if inward else anchor['speed'] >= target):
                    break
                middle = call((anchor['point']['rho']+boundary['point']['rho'])/2)
                if middle['status'] != 'accepted':
                    break

        contours = {}
        for level in LEVELS:
            ordered = sorted(rows)
            exact = [r for r in rows.values() if r['status'] == 'accepted'
                     and abs(r['speed']/level-1) < 1e-12]
            other_crossings = [(lo,hi) for lo,hi in zip(ordered[:-1],ordered[1:])
                if rows[lo]['status']==rows[hi]['status']=='accepted'
                and (rows[lo]['speed']-level)*(rows[hi]['speed']-level)<0]
            if len(exact) == 1 and not other_crossings:
                best = exact[0]
                rho = best['point']['rho']
                contours[level] = dict(status='resolved', point=best['point'], speed=best['speed'],
                    relative_speed_error=abs(best['speed']/level-1), rho_bracket=(rho,rho),
                    record_key=best['record_key'])
                continue
            brackets = [(lo, hi) for lo, hi in zip(ordered[:-1], ordered[1:])
                if rows[lo]['status'] == rows[hi]['status'] == 'accepted'
                and (rows[lo]['speed']-level)*(rows[hi]['speed']-level) <= 0]
            if not brackets:
                last = rows[max(rows)]
                accepted = [r for r in rows.values() if r['status'] == 'accepted']
                status = ('outside_window' if last['status'] == 'accepted'
                          and accepted and max(r['speed'] for r in accepted) < level
                          and not any(r['status']=='failed' for r in rows.values()) else 'unresolved')
                contours[level] = dict(status=status)
                continue
            if len(brackets) != 1:
                contours[level] = dict(status='unresolved', message='Multiple radial crossings')
                continue
            lo, hi = brackets[0]
            best = min((rows[lo], rows[hi]), key=lambda r: abs(r['speed']/level-1))
            for _ in range(40):
                if abs(best['speed']/level-1) < 1e-3 and hi-lo <= .001:
                    break
                f = np.clip((level-rows[lo]['speed'])/(rows[hi]['speed']-rows[lo]['speed']), .2, .8)
                mid = lo + f*(hi-lo)
                middle = call(mid)
                if middle['status'] != 'accepted':
                    best = None
                    break
                if abs(middle['speed']/level-1) < abs(best['speed']/level-1):
                    best = middle
                if (middle['speed']-level)*(rows[lo]['speed']-level) > 0:
                    lo = mid
                else:
                    hi = mid
            if best is None or abs(best['speed']/level-1) >= 1e-3 or hi-lo > .001:
                contours[level] = dict(status='unresolved')
            else:
                contours[level] = dict(status='resolved', point=best['point'], speed=best['speed'],
                    relative_speed_error=abs(best['speed']/level-1), rho_bracket=(lo, hi),
                    record_key=best['record_key'])
        return dict(method=method, theta_rad=theta, contours=contours, boundaries=boundaries,
                    records=[r['record_key'] for r in rows.values()],
                    boundary_records=boundary_records,
                    failures=[r['record_key'] for r in rows.values() if r['status'] == 'failed'] + boundary_failures)


def interpolation_error(left, middle, right):
    """Geometric contour interpolation check in coordinates normalized to the window."""
    errors = []
    transition = False
    for level in LEVELS:
        a, b, c = (r['contours'][level] for r in (left, middle, right))
        if all(v['status'] == 'resolved' for v in (a, b, c)):
            errors.append(float(np.linalg.norm(normalized_xy(b['point']) -
                (normalized_xy(a['point'])+normalized_xy(c['point']))/2)))
        elif len({v['status'] for v in (a, b, c)}) > 1:
            transition = True
    for name in ('stable_matter', 'composition_ceiling'):
        group = [r['boundaries'].get(name) for r in (left, middle, right)]
        if any(v and not v['resolved'] for v in group):
            errors.append(float('inf'))
        if any(group) and not all(group):
            transition = True
        if all(group):
            errors.append(float(np.linalg.norm(normalized_xy(group[1]['point']) -
                (normalized_xy(group[0]['point'])+normalized_xy(group[2]['point']))/2)))
    return max(errors, default=0.), transition


def make_parser(default_workers, default_output):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=default_workers)
    parser.add_argument('--method', choices=['both', *METHODS], default='both')
    parser.add_argument('--method-timeout', type=float, default=120.)
    parser.add_argument('--path-rtol', type=float, default=1e-6)
    parser.add_argument('--xi', type=float, default=-.85)
    parser.add_argument('--angular-rounds', type=int, default=6)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output', type=Path, default=default_output)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    return parser


def main(default_workers=4, default_output=None):
    args = make_parser(default_workers, default_output).parse_args()
    if args.workers < 1 or args.method_timeout <= 0 or args.path_rtol <= 0 or args.angular_rounds < 1:
        raise ValueError('Positive workers, timeout, tolerance and angular rounds required')
    if not np.isfinite(args.xi) or not -1 < args.xi < 1:
        raise ValueError('Use a finite sharpness factor strictly between -1 and 1')
    seed = json.loads(args.input.read_text())
    config = dict(T_0plus_MeV=0., xi=args.xi, ms_MeV=0., NM_type='PNM',
        B14_MeV=seed['calibration']['B14_MeV'], n0_MeV3=seed['calibration']['n0_MeV3'],
        path_rtol=args.path_rtol)
    sources = [HERE/'cold_scan.py', HERE/'cold_physics.py', HERE/'bounded_call.py']
    sources += sorted((ROOT/'RMFsolver').glob('*.py'))
    identity = dict(format='cold_adiabatic_contours_v1', physics=config,
        code_sha256={str(p.relative_to(ROOT)):sha256(p) for p in sources},
        input_sha256=sha256(args.input), method_timeout=args.method_timeout,
        angular_rounds=args.angular_rounds, smoke=args.smoke)
    output = args.output
    if args.smoke:
        output = output.with_name(output.stem+'-smoke.npy')
    methods = list(METHODS) if args.method == 'both' else [args.method]
    if args.dry_run:
        print(json.dumps(dict(output=str(output), workers=args.workers, methods=methods,
                             identity=identity, seed_coordinates=len(seed['coordinates'])), indent=2))
        return 0
    payload = dict(identity=identity, methods={}, angular_checks=[], complete=False,
                   record_directory=output.stem+'-records', levels_m_s=LEVELS,
                   calibration=seed['calibration'], reference_temperature_MeV=.1)
    if output.exists():
        payload = np.load(output, allow_pickle=True).item()
        compatible(payload, identity)
    scan = Scan(output, identity, seed['coordinates'], args.method_timeout)
    previous_sigint = signal.getsignal(signal.SIGINT)
    def interrupt(signum, frame):
        # Set this before ThreadPoolExecutor.__exit__ starts waiting.
        scan.cancel.set()
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, interrupt)
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            if args.smoke:
                tasks = [(m, 0., (3.5-1)/4.5) for m in methods]
                tasks += [(m, np.arctan2(30/120, (3.5-1)/4.5),
                           np.hypot(30/120, (3.5-1)/4.5)) for m in methods]
                rows = list(pool.map(lambda task: scan.evaluate(*task), tasks))
                payload['smoke_records'] = rows
                payload['complete'] = all(r['status']=='accepted' for r in rows)
                atomic_save(output, payload)
                print([(r['method'], r['status'], r['speed']) for r in rows], flush=True)
                return 0 if payload['complete'] else 2

            def run_rays(tasks):
                futures = {pool.submit(scan.trace_ray, m, t):(m, t) for m, t in tasks}
                for future in as_completed(futures):
                    m, t = futures[future]
                    row = future.result()
                    payload['methods'].setdefault(m, {})[t] = row
                    atomic_save(output, payload)
                    resolved = sum(r['status']=='resolved' for r in row['contours'].values())
                    print(f'{m}: angle {np.degrees(t):.3f}, {resolved}/7 levels, '
                          f'{len(row["failures"])} failed samples', flush=True)

            angles = sorted(set([p['theta_rad'] for p in seed['coordinates']] + [float(np.arctan2(115,120))]))
            run_rays([(m,t) for m in methods for t in angles if t not in payload['methods'].get(m,{})
                      or payload['methods'][m][t]['failures']])
            intervals = [(m,a,b,0) for m in methods for a,b in zip(angles[:-1],angles[1:])]
            pending = []
            for _ in range(args.angular_rounds):
                run_rays([(m,(a+b)/2) for m,a,b,d in intervals
                          if (a+b)/2 not in payload['methods'][m] or payload['methods'][m][(a+b)/2]['failures']])
                pending = []
                for m,a,b,depth in intervals:
                    middle = (a+b)/2
                    rays = payload['methods'][m]
                    error, transition = interpolation_error(rays[a],rays[middle],rays[b])
                    needs = error > .0025 or (transition and b-a > .0025)
                    payload['angular_checks'].append(dict(method=m, lower=a, upper=b,
                        error=error, transition=transition, resolved=not needs, depth=depth))
                    if needs:
                        pending += [(m,a,middle,depth+1),(m,middle,b,depth+1)]
                intervals = pending
                atomic_save(output, payload)
                if not intervals:
                    break
            payload['unresolved_angular_intervals'] = pending
            summary = {}
            for m in methods:
                rays = list(payload['methods'][m].values())
                summary[m] = dict(rays=len(rays), failures=sum(len(r['failures']) for r in rays),
                    unresolved=sum(c['status']=='unresolved' for r in rays for c in r['contours'].values()),
                    unresolved_boundaries=sum(not b['resolved'] for r in rays for b in r['boundaries'].values()),
                    supported_levels=[v for v in LEVELS if any(r['contours'][v]['status']=='resolved' for r in rays)],
                    outside_window_levels=[v for v in LEVELS if all(r['contours'][v]['status']=='outside_window' for r in rays)])
            payload.update(coverage=summary, complete=not pending and all(
                not r['failures'] and not r['unresolved'] and not r['unresolved_boundaries'] for r in summary.values()),
                runtime_this_invocation_s=time.monotonic()-scan.started)
            atomic_save(output, payload)
            print(json.dumps(summary, indent=2), flush=True)
            return 0 if payload['complete'] else 2
    except KeyboardInterrupt:
        scan.cancel.set()
        raise
    finally:
        signal.signal(signal.SIGINT, previous_sigint)


if __name__ == '__main__':
    raise SystemExit(main(default_output=HERE.parent/'local/adiabatic-T0plus0-xi-minus0p85.npy'))
