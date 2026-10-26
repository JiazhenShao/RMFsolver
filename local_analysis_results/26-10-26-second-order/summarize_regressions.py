"""Compare frozen before/after selections, separating time limits from failures."""
from pathlib import Path
import json

DIRECTORY=Path(__file__).resolve().parent
def read(name):
    data=json.loads((DIRECTORY/(name+'.json')).read_text())
    assert data.get('finished_epoch'),name+' has not finished'
    return data
def status(row):
    return 'TIMEOUT' if row['error'] and 'TestTimeout: Test exceeded' in row['error'] else row['status']
groups={}
for group in ('historical_60','analytic_slow','slow_bvp'):
    before=read('baseline_'+group)
    after=read('after_'+group)
    assert before['selection']==after['selection']
    a={row['id']:row for row in before['records']}
    b={row['id']:row for row in after['records']}
    assert a.keys()==b.keys()
    differences=[]
    for key in a:
        if status(a[key])!=status(b[key]):
            differences.append(dict(test=key,before=status(a[key]),after=status(b[key])))
    groups[group]=dict(before={k:before[k] for k in ('tests_run','passed','failed','errors')},
        after={k:after[k] for k in ('tests_run','passed','failed','errors')},
        status_differences=differences,
        before_source_sha256=before['source_sha256']['RMFsolver/phase_velocity.py'],
        after_source_sha256=after['source_sha256']['RMFsolver/phase_velocity.py'])
    if group!='slow_bvp':
        assert not differences
    else:
        for difference in differences:
            assert difference['before']=='PASS' and difference['after']=='TIMEOUT'
retries={}
for phase in ('baseline_600_slow_bvp','after_600_slow_bvp','after_600_warm_slow_bvp'):
    data=read(phase)
    retries[phase]=[dict(test=row['id'],status=status(row),seconds=row['seconds']) for row in data['records']]
assert [(r['test'],r['status']) for r in retries['baseline_600_slow_bvp']]==[
    (r['test'],r['status']) for r in retries['after_600_slow_bvp']]
for difference in groups['slow_bvp']['status_differences']:
    assert any(r['test']==difference['test'] and r['status']=='PASS'
               for r in retries['after_600_warm_slow_bvp'])
report=dict(groups=groups,retries=retries,new_assertion_failures=False,
    limitation='Cold tests that time out at both bounds remain incomplete; no claim of a complete passing BVP suite.')
(DIRECTORY/'regression-comparison.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
