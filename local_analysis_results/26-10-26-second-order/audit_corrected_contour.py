"""Read-only audit of new data, with all reports confined to this directory."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[1]
RUN = ROOT/'new_paper_calculations/26-10-26'
sys.path.insert(0,str(RUN/'shared'))
from cold_physics import validate_result

path = RUN/'local/adiabatic-T0plus0-xi-minus0p85-second-order.npy'
archive = DIRECTORY/'notebook-refresh/adiabatic-T0plus0-xi-minus0p85-leading-density.npy'
old = np.load(archive if archive.exists() else RUN/'local/adiabatic-T0plus0-xi-minus0p85.npy',
              allow_pickle=True).item()
data = np.load(path,allow_pickle=True).item()
assert data['complete'] and not data['unresolved_angular_intervals']
assert list(data['methods']) == ['analytic']
assert data['identity']['physics']['T_0plus_MeV'] == 0
old_hash = old['identity']['code_sha256']['RMFsolver/phase_velocity.py']
new_hash = data['identity']['code_sha256']['RMFsolver/phase_velocity.py']
assert old_hash != new_hash
for name,digest in data['identity']['code_sha256'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest,name
counts, maxima, residuals = Counter(),{},{}
records={}
min_temperature=np.inf
min_imbalance=np.inf
for file in path.with_name(data['record_directory']).glob('*.npy'):
    row=np.load(file,allow_pickle=True).item()
    assert row['identity'] == data['identity']
    assert row['method'] == 'analytic'
    assert 'calculation_identity' not in row and 'reuse_verification' not in row
    counts[row.get('evaluation','speed')+'/'+row['status']] += 1
    if row['status'] == 'accepted':
        result=row['result']
        validate_result(result)
        assert result['eos_approximation'] == 'quadratic_pressure_second_order_density'
        assert result['conservation_eos'] == result['eos_approximation']
        for name,value in result['quadratic_endpoint_residuals'].items():
            assert abs(value) <= 1e-10
            residuals[name]=max(residuals.get(name,0),abs(float(value)))
        for name,value in result['conservation_errors'].items():
            maxima[name]=max(maxima.get(name,0),float(value))
        if row['speed'] >= 50:
            min_temperature=min(min_temperature,result['T_inf'])
            H=result['h_0minus']/result['nB_0minus']
            min_imbalance=min(min_imbalance,(H-result['muB_inf'])/H)
    records[row['record_key']]=row
max_level=max_bracket=0.
for ray in data['methods']['analytic'].values():
    assert not ray['failures']
    for level,contour in ray['contours'].items():
        assert contour['status'] in ('resolved','outside_window')
        if contour['status'] == 'resolved':
            row=records[contour['record_key']]
            assert row['status']=='accepted'
            error=abs(row['speed']/level-1)
            width=contour['rho_bracket'][1]-contour['rho_bracket'][0]
            assert error<1e-3 and width<=.001
            max_level=max(max_level,error)
            max_bracket=max(max_bracket,width)
    assert all(b['resolved'] for b in ray['boundaries'].values())
    ordered=sorted((records[k] for k in ray['records']),key=lambda r:r['point']['rho'])
    for left,right in zip(ordered[:-1],ordered[1:]):
        if left['status']==right['status']=='accepted':
            assert right['speed'] >= left['speed']*(1-1e-8)
angular=[row['error'] for row in data['angular_checks'] if row['resolved']]
assert angular and max(angular)<=.0025
report=dict(passed=True,output=str(path.relative_to(ROOT)),coverage=data['coverage'],
    counts=dict(counts),max_conservation_errors=maxima,max_endpoint_residuals=residuals,
    max_level_relative_error=max_level,max_radial_bracket=max_bracket,
    max_accepted_angular_error=max(angular),minimum_T_inf_at_speed_ge_50=float(min_temperature),
    minimum_relative_heat_gap_at_speed_ge_50=float(min_imbalance),
    old_phase_velocity_sha256=old_hash,new_phase_velocity_sha256=new_hash,
    source_hashes_match=True,old_record_reuse=False,
    runtime_s=data['runtime_this_invocation_s'])
(DIRECTORY/'contour-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
