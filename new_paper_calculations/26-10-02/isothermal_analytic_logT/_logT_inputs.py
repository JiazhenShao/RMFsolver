"""Static inputs: no numerical velocities, profiles or warm contour reuse."""
import argparse
import json
from pathlib import Path
import numpy as np
from _logT_common import PHYSICS, TEMPERATURES, load, sha256, atomic_save


def static_physics():
    return {k:v for k,v in PHYSICS.items() if k!='xi'}


def load_inputs(directory):
    directory = Path(directory)
    manifest = json.loads((directory/'inputs/manifest.json').read_text())
    if manifest.get('schema')!=2 or manifest.get('use')!='static_boundaries_only':
        raise ValueError('Use the prepared static input bundle, not the copied numerical inputs')
    for name, digest in manifest['files'].items():
        if sha256(directory/name)!=digest:
            raise ValueError(f'Input hash mismatch: {name}')
    if manifest['physics']!=static_physics():
        raise ValueError('Static boundary physics mismatch')
    bundle = load(directory/'inputs/static-boundaries.npy')
    if bundle['domain_fingerprint']!=manifest['fingerprint']:
        raise ValueError('Domain fingerprint mismatch')
    cache=dict(bundle['boundary_cache'])
    # geomspace may differ by a final floating-point bit across NumPy builds.
    # Alias only roundoff-equivalent temperatures; never interpolate a root.
    for T in TEMPERATURES:
        if float(T).hex() not in cache:
            row=next((r for r in bundle['boundary_cache'].values()
                      if np.isclose(T,r['T_0minus'],rtol=2.e-14,atol=0.)),None)
            if row is not None:
                cache[float(T).hex()]=dict(row,T_0minus=float(T),source_temperature=row['T_0minus'])
    return dict(n0=bundle['n0'], fingerprint=bundle['domain_fingerprint'],
                boundary_geometry=bundle['boundary_geometry'], boundary_cache=cache,
                manifest=manifest)


def prepare_static_inputs(source, directory):
    """One-time extraction from the downloaded run; no solver calls."""
    source, directory = Path(source), Path(directory)
    checkpoint = load(source/'isothermal-logT.npy')
    if {k:v for k,v in checkpoint['identity']['physics'].items() if k!='xi'}!=static_physics():
        raise ValueError('Downloaded boundary physics differs')
    domain = load(source/'inputs/isothermal-domain.npy')
    if domain['domain_fingerprint']!=checkpoint['domain_fingerprint']:
        raise ValueError('Downloaded boundary provenance differs')
    n0 = float(np.nanmedian(domain['nB_0minus_grid']/domain['nB_0minus_over_n0_grid']))
    with np.load(source/'inputs/isothermal-numerical-contour.npz',allow_pickle=False) as archive:
        geometry = {key:archive[key].copy() for key in ('inner_boundary','outer_boundary')}
    cache = {}
    for key,row in checkpoint['boundaries'].items():
        if (row.get('status')=='success' and 1<=row['lower']<row['upper']<=5.5
                and max(abs(v) for v in row['residuals_MeV'].values())<=1.e-5):
            cache[key] = dict(row, lower_clipped=False, upper_clipped=False,
                              source='validated_static_boundary_snapshot')
    bundle = dict(n0=n0, domain_fingerprint=domain['domain_fingerprint'],
                  boundary_geometry=geometry, boundary_cache=cache)
    destination = directory/'inputs/static-boundaries.npy'
    if destination.exists() or (directory/'inputs/manifest.json').exists():
        raise FileExistsError('Refusing to overwrite prepared static inputs')
    atomic_save(destination,bundle)
    manifest = dict(schema=2,use='static_boundaries_only', physics=static_physics(),
        fingerprint=domain['domain_fingerprint'],
        files={'inputs/static-boundaries.npy':sha256(destination)},
        source_checkpoint_sha256=sha256(source/'isothermal-logT.npy'),
        source_domain_sha256=sha256(source/'inputs/isothermal-domain.npy'),
        source_boundary_geometry_sha256=sha256(source/'inputs/isothermal-numerical-contour.npz'),
        source_xi_ignored_for_static_boundaries=checkpoint['identity']['physics']['xi'])
    (directory/'inputs/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return load_inputs(directory)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--directory',type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args()
    prepared=prepare_static_inputs(args.source,args.directory)
    print(f"Prepared {len(prepared['boundary_cache'])} validated static boundary temperatures; no velocity results copied")
