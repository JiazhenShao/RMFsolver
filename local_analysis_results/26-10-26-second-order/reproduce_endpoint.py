from pathlib import Path
import json,sys,time
import numpy as np
from scipy.optimize import brentq
from RMFsolver import phase_velocity as pv,constants as const
ROOT=Path(__file__).resolve().parents[2]
N0=.16*const.MeV_fm**3; BAG=189.1565957288247
STATES=((8.860992751744538,3.0268624251597),(27.83373842080244,2.96875),(70.6735943750199,1.434488038438507),(10.15113868209788,4.500181704610503))
rows=[]
for T,n in STATES:
 old=pv.analytic_velocity_adiabatic_slow(T,n*N0,BAG,T_0plus=0,xi=-.85)
 semi=pv.semi_analytic_velocity_adiabatic_slow(T,n*N0,BAG,T_0plus=0,xi=-.85)
 assert old['success'],old['message'];assert semi['success'],semi['message']
 H=old['h_0minus']/(n*N0);C=108*np.pi**2*(old['P_0minus']+BAG**4)
 mu=H*brentq(lambda x:x**4-2*x**3+C/H**4,0,C**.25/H,xtol=1e-15)
 Ti=np.sqrt(mu*(H-mu)/(9*np.pi**2));A=3/np.sqrt(2)*np.sqrt((H-mu)/mu)
 weak=pv._weak_rate_from_mu_q(mu/3)
 I2=9*np.pi**(7/3)*weak['gamma']/(8*np.sqrt(2)*pv._TRANSPORT_H_CONST*pv._TRANSPORT_ALPHA_S**(5/3)*mu)*A**(7/3)*(24/7)
 speed=np.sqrt(2*I2/(old['lambda_n']**2*(1-A)*(1-.85*A)))*299792458.
 n2=mu**3/(27*np.pi**2)+mu*Ti**2/3
 row=dict(T_0minus=T,nB_0minus_over_n0=n,upstream_H=H,C=C,old={k:old[k] for k in ('muB_inf','T_inf','A_adb','velocity_m_s','jB','I2','nB_inf','lambda_n')},independent_second_order=dict(muB_inf=mu,T_inf=Ti,A_adb=A,velocity_m_s=speed,I2=I2,nB_inf_second_order=n2,unexpanded_ratio_residual=4*(old['P_0minus']+BAG**4)/n2/H-1),semi={k:semi[k] for k in ('muB_inf','T_inf','a_0plus','velocity_m_s','jB','I2','nB_inf','lambda_n')})
 rows.append(row);print(json.dumps(row),flush=True)
reference=pv.analytic_velocity_adiabatic_slow(10,3*N0,180,T_0plus=5,xi=-.5)
output=dict(states=rows,old_reference_B180={k:reference[k] for k in ('muB_inf','T_inf','A_adb','a_0plus','jB','I2')})
(ROOT/'local_analysis_results/26-10-26-second-order/endpoint-before-and-independent-targets.json').write_text(json.dumps(output,indent=2)+'\n')
