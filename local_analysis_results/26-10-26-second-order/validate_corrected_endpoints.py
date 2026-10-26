"""Reproduce the corrected table and matched exact-cold spatial comparisons."""
from pathlib import Path
import json
import signal
import time
import numpy as np
from RMFsolver import phase_velocity as pv, constants as const

DIRECTORY = Path(__file__).resolve().parent
baseline = json.loads((DIRECTORY/'endpoint-before-and-independent-targets.json').read_text())
rows = []
class Deadline(BaseException):
    pass
def alarm(signum, frame):
    raise Deadline('Matched cold BVP exceeded 120 seconds')
signal.signal(signal.SIGALRM, alarm)
for index, before in enumerate(baseline['states']):
    T, density = before['T_0minus'], before['nB_0minus_over_n0']
    nB = density*.16*const.MeV_fm**3
    result = pv.analytic_velocity_adiabatic_slow(T,nB,189.1565957288247,T_0plus=0,xi=-.85)
    assert result['success'],result['message']
    for key in ('muB_inf','T_inf','A_adb','velocity_m_s','I2'):
        np.testing.assert_allclose(result[key],before['independent_second_order'][key],rtol=1e-12)
    row = dict(T_0minus=T,nB_0minus_over_n0=density,old=before['old'],semi=before['semi'],
               corrected=result,ratio_to_semi=result['velocity_m_s']/before['semi']['velocity_m_s'])
    if index < 3:
        started = time.monotonic()
        signal.setitimer(signal.ITIMER_REAL,120)
        try:
            bvp = pv.solve_front_adiabatic_slow(T,nB,189.1565957288247,T_0plus=0,
                jB_guess=before['semi']['jB'],return_profile=True)
            row['bvp']={k:v for k,v in bvp.items() if isinstance(v,(str,bool,float,int,type(None),dict))}
            row['bvp']['velocity_m_s']=bvp.get('u_0minus',np.nan)*299792458.
            if bvp['success']:
                assert max(bvp['conservation_errors'].values()) < 1e-8
                assert bvp['T'][0] == 0
                np.testing.assert_allclose(bvp['velocity_m_s'] if 'velocity_m_s' in bvp
                    else bvp['u_0minus']*299792458.,(34.89,81.63,250.42)[index],rtol=3e-4)
        except Deadline as exc:
            row['bvp']=dict(success=False,message=str(exc))
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
            row['bvp']['runtime_s']=time.monotonic()-started
    else:
        row['bvp']=dict(success=None,message='Not rerun: user supplied failure at this state')
    rows.append(row)
    (DIRECTORY/'endpoint-after-validation.json').write_text(json.dumps(dict(states=rows),indent=2)+'\n')
    print(index,result['velocity_m_s'],row['bvp'].get('velocity_m_s'),row['bvp'].get('success'),flush=True)
