"""Reuse the first high-speed run to add 200000, 500000, and 1000000 m/s.

This configures the same bounded runner, not a different physics method. Its
separate seed cache avoids recomputing accepted 10000--100000 m/s evaluations.
"""
import copy
import hashlib
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def seed_cache(original, extension):
    cached = copy.deepcopy(original)
    for evaluation in extension['evaluations']:
        row = evaluation.get('root')
        if row and row['result'].get('validated') and row['result'].get('success'):
            key = f"{float(row['T_0minus']).hex()}:{float(row['nB_0minus_over_n0']).hex()}"
            cached['points'].setdefault(key, copy.deepcopy(row))
    return cached


def main():
    import extend_isothermal_levels as runner
    sys.path.insert(0, str(HERE / 'isothermal_analytic_logT'))
    from _logT_common import atomic_save
    extension_path = runner.OUTPUT
    extension = np.load(extension_path, allow_pickle=True).item()
    if any(row['status'] == 'pending' for row in extension['targets'].values()):
        raise ValueError('Finish the first high-speed run before snapshotting its cache')
    original = np.load(runner.BASELINE, allow_pickle=True).item()
    for relative, digest in extension['identity']['input_hashes'].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != digest:
            raise ValueError('First extension input hash changed')
    cached = seed_cache(original, extension)
    cached['seed_provenance'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (runner.BASELINE, extension_path)}
    seed_path = extension_path.with_name('isothermal-high-speed-seed-cache.npy')
    if seed_path.exists():
        if np.load(seed_path, allow_pickle=True).item()['seed_provenance'] != cached['seed_provenance']:
            raise ValueError('Preserve the existing cache: its source hashes changed')
    else:
        atomic_save(seed_path, cached)
    runner.BASELINE = seed_path
    runner.OUTPUT = extension_path.with_name('isothermal-outer-speed-extension.npy')
    runner.LEVELS = (200000., 500000., 1000000.)
    runner.main()


if __name__ == '__main__':
    main()
