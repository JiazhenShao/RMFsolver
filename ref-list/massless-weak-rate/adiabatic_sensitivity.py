#!/usr/bin/env python3
"""Cold-interface adiabatic EOS trajectories and diffusion-weighted weak rates.

No BVP or velocity eigenvalue is solved. All energies use MeV. The upstream
QMC-RMF3 calculation is isolated in adiabatic_upstream.py; this numerical core
needs only NumPy. The same quark density and pressure are used for all rates.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from functools import lru_cache
import json
import math
from pathlib import Path
import time

import numpy as np

from cold_sympy import cold_phase_space
from thermal_sympy import phase_space_t2
from validate_rate import (HERE, PREF, _bisect, _chemical_from_density,
                           eq23_rate, exact_rate, gauss, number_density)

ALPHA_S = .3
H_D = 1.81317
COMPOSITIONS = (.005,.02,.05,.075,.1,.146,.2,.25,.3,.35,.4,.45,.5,
                .55,.6,.65,.7,.75,.8,.85,.9,.95,.98,.99,.995)
MODEL_NAMES = ('exact', 'eq23', 'sommerfeld_T2')
OUTER_EDGES = (0., .125, .5, .875, 1.)


def pressure_kinetic(chemical, temperature):
    """Quarks plus thermal gluons; the constant bag pressure cancels."""
    mu = np.asarray(chemical)
    return float(np.sum(mu**4/(4*math.pi**2)+mu**2*temperature**2/2)
                 +19*math.pi**2*temperature**4/36)


def diffusion(muB, temperature):
    if temperature <= 0:
        raise ValueError('Use the analytic integrand limit at the cold endpoint')
    qD = math.sqrt(6*ALPHA_S/math.pi)*muB/3
    return 1/(24*ALPHA_S**2/math.pi*(H_D*temperature**(5/3)/qD**(2/3)
                  +math.pi**3*temperature**2/(12*qD)))


class ColdInterfaceTrajectory:
    """Constant nB and pressure from a cold interface to flavor equilibrium.

    cold_mu=(pi^2*nB)^(1/3) is a numerical energy scale, not constant muB.
    z=(T/T_inf)^(1/3) regularizes the cold diffusion singularity.
    """
    def __init__(self, a_0plus, cold_mu=400., root_iterations=80):
        if not 0 < a_0plus < 1 or cold_mu <= 0:
            raise ValueError('Require 0 < a_0plus < 1 and positive density scale')
        self.a_0plus = float(a_0plus)
        self.cold_mu = float(cold_mu)
        self.nB = self.cold_mu**3/math.pi**2
        self.root_iterations = root_iterations
        self.cold_chemical = self.chemical(self.a_0plus, 0.)
        self.pressure = pressure_kinetic(self.cold_chemical, 0.)
        self._pressure_scaled = self.pressure/self.cold_mu**4
        self.T_inf = self.cold_mu*_bisect(
            lambda t: self._pressure(0.,t)-self._pressure_scaled,
            0., 1., iterations=root_iterations)

    @staticmethod
    def _chemical(a, t):
        chemical=[]
        for factor in (1.,1+a,1-a):
            mu=float(np.cbrt(factor))
            if t < .1*mu:
                # Near cold, Newton from the cold root avoids loss of digits
                # in sinh(asinh(...)/3), without flooring the temperature.
                for _ in range(4):
                    mu-=(mu**3+math.pi**2*t*t*mu-factor)/(3*mu*mu+math.pi**2*t*t)
            else:
                mu=_chemical_from_density(factor/math.pi**2,t)
            chemical.append(mu)
        return tuple(chemical)

    def chemical(self, a, temperature):
        return tuple(self.cold_mu*mu for mu in self._chemical(a,temperature/self.cold_mu))

    def _pressure(self, a, t):
        return pressure_kinetic(self._chemical(a,t), t)

    @lru_cache(maxsize=2048)
    def state(self, z):
        if not 0 <= z <= 1:
            raise ValueError('z must lie in [0,1]')
        temperature = self.T_inf*z**3
        t = temperature/self.cold_mu
        if z == 0:
            a = self.a_0plus
        elif z == 1:
            a = 0.
        else:
            upper_residual = self._pressure(self.a_0plus,t)-self._pressure_scaled
            if -64*np.finfo(float).eps*self._pressure_scaled <= upper_residual <= 0:
                # The physical thermal displacement is below pressure roundoff.
                # Keep the finite temperature and analytic Jacobian; no T floor.
                a = self.a_0plus
            else:
                a = _bisect(lambda a: self._pressure(a,t)-self._pressure_scaled,
                            0., self.a_0plus, iterations=self.root_iterations)
        mu = np.array(self._chemical(a,t))
        density = np.array([1.,1+a,1-a])/math.pi**2
        chi = 3*mu**2/math.pi**2+t**2
        pressure_a = (density[1]/chi[1]-density[2]/chi[2])/math.pi**2
        pressure_t = float(np.sum(mu**2*t-2*mu*t*density/chi)+19*math.pi**2*t**3/9)
        minus_da_dz = (pressure_t/pressure_a*3*self.T_inf/self.cold_mu*z**2
                       if 0 < z < 1 else (0. if z == 0 else math.inf))
        chemical = tuple(self.cold_mu*mu)
        return dict(a=float(a), T=temperature, chemical=chemical,
                    muB=float(sum(chemical)), muK=float(chemical[1]-chemical[2]),
                    minus_da_dz=float(minus_da_dz),
                    pressure_relative_residual=abs(self._pressure(a,t)/self._pressure_scaled-1))


def three_rates(chemical, temperature, phase_order=16, tail=48):
    return np.array([exact_rate(*chemical,temperature,n=phase_order,tail=tail),
                     eq23_rate(*chemical,temperature),
                     PREF*(cold_phase_space(*chemical)+temperature**2*phase_space_t2(*chemical))])


def cold_endpoint_integrand(trajectory):
    """Finite limit of D*Gamma/nB*abs(da/dz), z=(T/Tinf)^(1/3)."""
    mu = np.array(trajectory.cold_chemical)
    qD = math.sqrt(6*ALPHA_S/math.pi)*sum(mu)/3
    diffusion_leading = math.pi*qD**(2/3)/(24*ALPHA_S**2*H_D)
    coefficient = (3*diffusion_leading*trajectory.T_inf**(1/3)*sum(mu**2)
                   /(trajectory.nB**2*(mu[1]-mu[2])))
    cold = PREF*cold_phase_space(*mu)
    return coefficient*np.array([cold, eq23_rate(*mu,0.), cold])


def weighted_integrand(trajectory,z,phase_order=16,tail=48):
    if z == 0:
        return cold_endpoint_integrand(trajectory)
    if z == 1:
        # The exact endpoint has Gamma=0 and an infinite a-Jacobian.
        # Plot a one-sided approximation; quadrature never samples it.
        z = 1-1e-7
    state = trajectory.state(float(z))
    return (diffusion(state['muB'],state['T'])*state['minus_da_dz']/trajectory.nB
            *three_rates(state['chemical'],state['T'],phase_order,tail))


def integrated_estimates(trajectory,outer_order=12,phase_order=16,tail=48,
                         diffusion_scale=1.):
    nodes,weights = gauss(outer_order)
    total,constant = np.zeros(3),np.zeros(3)
    max_residual = 0.
    records=[]
    for lo,hi in zip(OUTER_EDGES[:-1],OUTER_EDGES[1:]):
        half=(hi-lo)/2
        for z,w in zip((hi+lo)/2+half*nodes,half*weights):
            state=trajectory.state(float(z))
            rates=three_rates(state['chemical'],state['T'],phase_order,tail)
            unweighted=state['minus_da_dz']/trajectory.nB*rates
            weighted=diffusion_scale*diffusion(state['muB'],state['T'])*unweighted
            total+=w*weighted
            constant+=w*unweighted
            max_residual=max(max_residual,state['pressure_relative_residual'])
            records.append(dict(z=float(z),a=state['a'],T=state['T'],muB=state['muB'],
                       muK=state['muK'],D_K=diffusion(state['muB'],state['T']),
                       mu_u=state['chemical'][0],mu_d=state['chemical'][1],mu_s=state['chemical'][2],
                       quadrature_weight=float(w),
                       **{f'weighted_{name}':float(weighted[i]) for i,name in enumerate(MODEL_NAMES)},
                       **{f'rate_{name}':float(rates[i]) for i,name in enumerate(MODEL_NAMES)}))
    if np.any(total<=0) or not np.all(np.isfinite(total)):
        raise ArithmeticError('Nonpositive or nonfinite weighted rate integral')
    return dict(a_0plus=trajectory.a_0plus,cold_mu=trajectory.cold_mu,T_inf=trajectory.T_inf,
                nB=trajectory.nB,integrals=total.tolist(),ratios=np.sqrt(total/total[0]).tolist(),
                constant_D_integrals=constant.tolist(),
                constant_D_ratios=np.sqrt(constant/constant[0]).tolist(),
                max_pressure_relative_residual=max_residual,records=records)


def cutoff_check(trajectory,cutoff_fraction,phase_order=24,order=24):
    """Independent log-temperature quadrature plus leading analytic cold tail.

    The missing integral has leading order Tcut^(1/3); diffusion's second
    term makes the first omitted tail contribution order Tcut^(2/3).
    """
    nodes,weights=gauss(order)
    lower=math.log(cutoff_fraction)
    edges=np.linspace(lower,0.,9)
    integral=np.zeros(3)
    for lo,hi in zip(edges[:-1],edges[1:]):
        half=(hi-lo)/2
        for logt,w in zip((lo+hi)/2+half*nodes,half*weights):
            z=math.exp(logt/3)
            integral+=w*weighted_integrand(trajectory,z,phase_order)*z/3
    correction=cold_endpoint_integrand(trajectory)*cutoff_fraction**(1/3)
    return dict(cutoff_fraction=cutoff_fraction,raw_integrals=integral.tolist(),
                corrected_integrals=(integral+correction).tolist())


def write_csv(path,rows):
    if rows:
        with Path(path).open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=HERE/'results')
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--force',action='store_true')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    checkpoint=args.output/'adiabatic-quadrature.json'
    provenance=args.output/'adiabatic-quadrature-provenance.json'
    signature={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in
               ('adiabatic_sensitivity.py','validate_rate.py','cold_sympy.py','thermal_sympy.py')}
    if checkpoint.exists() and not args.force:
        if not provenance.exists() or json.loads(provenance.read_text())['source_sha256']!=signature:
            raise RuntimeError('Quadrature source signature changed; rerun with --force')
    provenance.write_text(json.dumps(dict(source_sha256=signature,numpy_version=np.__version__,
                                         alpha_s=ALPHA_S,H_D=H_D),indent=2)+'\n')
    data={} if args.force or not checkpoint.exists() else json.loads(checkpoint.read_text())
    compositions=(.146,.5,.9) if args.quick else COMPOSITIONS
    for a in compositions:
        trajectory=ColdInterfaceTrajectory(a)
        settings=[(8,12,48),(12,16,48),(18,16,48)]
        if a in (.146,.5,.9,.995):
            settings.extend([(18,28,48),(18,28,64)])
        if args.quick:
            settings=[(8,8,48),(12,12,48)]
        for outer,phase,tail in settings:
            key=f'{a:g}:{outer}:{phase}:{tail}'
            if key in data:
                continue
            start=time.monotonic()
            result=integrated_estimates(trajectory,outer,phase,tail)
            result.update(outer_order=outer,phase_order=phase,tail=tail,
                          wall_seconds=time.monotonic()-start)
            data[key]=result
            temporary=checkpoint.with_suffix('.tmp')
            temporary.write_text(json.dumps(data,indent=2)+'\n')
            temporary.replace(checkpoint)
            print(f'a={a:g} outer={outer} phase={phase} tail={tail}: '
                  f'ratios={result["ratios"]} ({result["wall_seconds"]:.1f}s)',flush=True)
    rows=[]
    for a in compositions:
        candidates=[value for value in data.values() if value['a_0plus']==a]
        final=max(candidates,key=lambda x:(x['outer_order'],x['phase_order'],x['tail']))
        row={k:final[k] for k in ('a_0plus','cold_mu','T_inf','nB','max_pressure_relative_residual')}
        for i,name in enumerate(MODEL_NAMES):
            row[f'I2_{name}']=final['integrals'][i]
            row[f'ratio_{name}']=final['ratios'][i]
            row[f'constant_D_ratio_{name}']=final['constant_D_ratios'][i]
        rows.append(row)
    write_csv(args.output/'adiabatic-universal.csv',rows)


if __name__=='__main__':
    main()
