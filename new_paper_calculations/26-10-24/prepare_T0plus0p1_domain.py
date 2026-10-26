#!/usr/bin/env python3
"""Copy the completed relativistic contour's grid and scalar flux seeds.

No EOS or analytical velocity calculation is performed. The historical domain
supplies geometry metadata only; every plotted velocity is recomputed later.
"""
import argparse
import copy
from pathlib import Path

import numpy as np
from _slow_front_adapter import sha256

SCRIPT_DIR = Path(__file__).resolve().parent


def prepare_domain():
    source_path = SCRIPT_DIR / 'inputs/relativistic-state-of-art.npy'
    legacy_path = SCRIPT_DIR / 'inputs/legacy-domain.npy'
    source = np.load(source_path, allow_pickle=True).item()
    manifest = copy.deepcopy(np.load(legacy_path, allow_pickle=True).item())
    if source['calibration'] != manifest['calibration'] or source['T_0plus_MeV'] != .1:
        raise ValueError('Reference calibration or interface temperature mismatch')
    if not np.all(source['plot_success_grid']):
        raise ValueError('Reference display grid is incomplete')
    legacy_points = {(p['theta_index'], p['radial_index']): p for p in manifest['points']}
    points = []
    for theta, radial in np.ndindex(source['plot_jB_grid'].shape):
        p = copy.deepcopy(legacy_points[theta, radial])
        fraction = float(source['plot_rho_grid'][theta, radial])
        state = p['states'][0]
        state.update(T_0minus_MeV=float(source['plot_T_0minus_MeV_grid'][theta, radial]),
                     nB_0minus_over_n0=float(source['plot_nB_0minus_over_n0_grid'][theta, radial]))
        state['rho'] = state['rho_inner_boundary'] + fraction * (state['rho_outer_boundary'] - state['rho_inner_boundary'])
        p.update(state, point_index=len(points), trajectory_index=len(points),
                 radial_fraction=fraction, radial_fraction_used=fraction,
                 relativistic_jB_guess=float(source['plot_jB_grid'][theta, radial]),
                 source_grid_index=(theta, radial))
        if not np.isfinite(p['relativistic_jB_guess']) or p['relativistic_jB_guess'] <= 0:
            raise ValueError('Invalid numerical reference flux')
        points.append(p)
    manifest.update(points=points, manifest_type='T0plus_slow_numerical_reference_grid_v1',
                    selection_rule='saved_relativistic_numerical_grid',
                    reference_source='26-08-24/T0plus0p1-uNmax-state-of-art-complete-5km.npy',
                    reference_source_sha256=sha256(source_path),
                    legacy_domain_sha256=sha256(legacy_path),
                    geometry_role='historical sampling grid; not a new-model admissibility boundary')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=SCRIPT_DIR / 'T0plus0p1-domain.npy')
    args = parser.parse_args()
    domain = prepare_domain()
    if args.output.exists():
        raise SystemExit('Output exists; the bundled prepared domain is ready to use.')
    np.save(args.output, domain, allow_pickle=True)
    print(f'Saved {len(domain["points"])} numerical reference targets to {args.output}')
