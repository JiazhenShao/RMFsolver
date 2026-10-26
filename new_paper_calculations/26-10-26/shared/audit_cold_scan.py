"""Audit the complete saved scan without rerunning EOS or front calculations."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from cold_physics import validate_result
from cold_scan import normalized_xy

ROOT = Path(__file__).resolve().parents[3]


def main():
    path = (Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/
        'new_paper_calculations/26-10-26/local/adiabatic-T0plus0-xi-minus0p85.npy')
    data = np.load(path, allow_pickle=True).item()
    assert data['complete'] and not data['unresolved_angular_intervals']
    assert data['identity']['physics']['T_0plus_MeV'] == 0
    for name, digest in data['identity']['code_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    record_dir = path.with_name(data['record_directory'])
    counts, max_errors = Counter(), {}
    records = {}
    for file in record_dir.glob('*.npy'):
        row = np.load(file, allow_pickle=True).item()
        assert row['identity'] == data['identity']
        counts[row['method']+'/'+row.get('evaluation','speed')+'/'+row['status']] += 1
        if 'calculation_identity' in row:
            original = row['calculation_identity']
            assert original['physics'] == data['identity']['physics']
            assert original['input_sha256'] == data['identity']['input_sha256']
            for name,digest in original['code_sha256'].items():
                if name.startswith('RMFsolver/'):
                    assert data['identity']['code_sha256'][name] == digest
            physics_name = 'new_paper_calculations/26-10-26/shared/cold_physics.py'
            source = (ROOT/physics_name).read_text()
            original_source = source[:source.index('\n\ndef classify')].rstrip()+'\n'
            assert original['code_sha256'][physics_name] in (
                hashlib.sha256(original_source.encode()).hexdigest(),
                data['identity']['code_sha256'][physics_name])
        if row['status']=='accepted':
            validate_result(row['result'])
            for name, value in row['result']['conservation_errors'].items():
                max_errors[name] = max(max_errors.get(name,0),float(value))
        records[row['record_key']] = row
    largest_residual, largest_bracket, nonmonotone = 0., 0., []
    for method,rays in data['methods'].items():
        for theta,ray in rays.items():
            assert not ray['failures']
            ordered = sorted((records[k] for k in ray['records']),key=lambda r:r['point']['rho'])
            for left,right in zip(ordered[:-1],ordered[1:]):
                if left['status']==right['status']=='accepted' and right['speed'] < left['speed']*(1-1e-8):
                    nonmonotone.append((method,float(theta),left['record_key'],right['record_key']))
            for level, contour in ray['contours'].items():
                assert contour['status'] in ('resolved','outside_window')
                if contour['status']=='resolved':
                    row = records[contour['record_key']]
                    assert row['status']=='accepted'
                    error = abs(row['speed']/level-1)
                    width = contour['rho_bracket'][1]-contour['rho_bracket'][0]
                    assert error < 1e-3 and width <= .001
                    largest_residual=max(largest_residual,error)
                    largest_bracket=max(largest_bracket,width)
            assert all(b['resolved'] for b in ray['boundaries'].values())
    # The current physical branch is sampled as a single radial crossing.
    # This also guards the exact-sample shortcut against a nonmonotone branch.
    assert not nonmonotone, nonmonotone[:3]
    terminal = [r['error'] for r in data['angular_checks'] if r['resolved']]
    assert terminal and max(terminal) <= .0025
    report = dict(passed=True, physical_settings=data['identity']['physics'],
        record_counts=dict(counts), coverage=data['coverage'],
        max_conservation_errors=max_errors, largest_level_relative_error=largest_residual,
        largest_radial_bracket=largest_bracket,
        max_accepted_angular_interpolation_error=max(terminal), nonmonotone_intervals=nonmonotone,
        source_hashes_match=True, reference_T_0plus_MeV=.1,
        runtime_last_invocation_s=data['runtime_this_invocation_s'])
    output=ROOT/'local_analysis_results/26-10-26/cold-contour-verification.json'
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
