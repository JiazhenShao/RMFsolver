#!/usr/bin/env python3
"""Audit repeated ratios without rescaling the production quadrature.

Use saved upstream pressure and enthalpy to reconstruct each dimensional
quark EOS independently in 40-digit arithmetic. Integrate in composition
with a=a_0plus*(1-y**6), without the production EOS or its Jacobian.
The microscopic rates and transport law are the same model under audit.
No production results are overwritten.
"""
import csv
import hashlib
import json
from pathlib import Path
import time

import mpmath as mp
import numpy as np

from adiabatic_sensitivity import three_rates, diffusion


def evaluate(state, bag, outer_order=18, phase_order=16):
    with mp.workdps(40):
        pressure_target = mp.mpf(str(state['P_0minus'])) + mp.mpf(str(bag))
        nB = 4*pressure_target/mp.mpf(str(state['h_over_nB_0minus']))

        def chemical(a, temperature):
            densities = [nB, nB*(1+a), nB*(1-a)]
            if temperature == 0:
                return [mp.root(mp.pi**2*density, 3) for density in densities]
            p = mp.pi**2*temperature**2/3
            return [2*mp.sqrt(p)*mp.sinh(mp.asinh(mp.pi**2*density/(2*p**mp.mpf('1.5')))/3)
                    for density in densities]

        def pressure(a, temperature):
            return (sum(mu**4/(4*mp.pi**2)+mu**2*temperature**2/2
                        for mu in chemical(a, temperature))
                    +19*mp.pi**2*temperature**4/36)

        def bisect(function, lower, upper):
            lower, upper = mp.mpf(lower), mp.mpf(upper)
            assert function(lower) <= 0 and function(upper) >= 0
            for _ in range(110):
                middle = (lower+upper)/2
                if function(middle) > 0:
                    upper = middle
                else:
                    lower = middle
            return (lower+upper)/2

        a_0plus = bisect(lambda a: pressure(a, 0)-pressure_target, 0, 1)
        temperature_upper = mp.root(mp.pi**2*nB, 3)
        T_inf = bisect(lambda temperature: pressure(0, temperature)-pressure_target,
                       0, temperature_upper)
        nodes, weights = np.polynomial.legendre.leggauss(outer_order)
        total = np.zeros(3)
        maximum_pressure_residual = mp.mpf(0)
        edges = (0, .125, .5, .875, 1)
        for lower, upper in zip(edges[:-1], edges[1:]):
            half = (upper-lower)/2
            for y, weight in zip((upper+lower)/2+half*nodes, half*weights):
                a = a_0plus*(1-mp.mpf(float(y))**6)
                temperature = bisect(lambda t: pressure(a, t)-pressure_target,
                                     0, temperature_upper)
                maximum_pressure_residual = max(maximum_pressure_residual,
                                                abs(pressure(a, temperature)/pressure_target-1))
                potentials = tuple(float(mu) for mu in chemical(a, temperature))
                rates = three_rates(potentials, float(temperature), phase_order, 48)
                total += (weight*6*float(a_0plus)*y**5/float(nB)
                          *diffusion(sum(potentials), float(temperature))*rates)
        return dict(T_0minus=state['T_0minus'],
                    nB_0minus_over_n0=state['nB_0minus_over_n0'],
                    target_a_0plus=state['target_a_0plus'], a_0plus=float(a_0plus),
                    nB=float(nB), T_inf=float(T_inf), integrals=total.tolist(),
                    ratios=np.sqrt(total/total[0]).tolist(),
                    max_pressure_relative_residual=float(maximum_pressure_residual),
                    outer_order=outer_order, phase_order=phase_order, tail=48)


def main():
    directory = Path(__file__).resolve().parent/'results'
    source = directory/'adiabatic-upstream.json'
    upstream = json.loads(source.read_text())
    selected = [state for state in upstream['states']
                if state['T_0minus'] in (0., 20., 100.)
                and state['target_a_0plus'] in (.9, .995)]
    assert len(selected) == 6
    report = dict(method='Independent dimensional EOS and a-coordinate quadrature; no cached integrals',
                  eos_decimal_digits=40, eos_bisection_iterations=110,
                  upstream_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  source_sha256={name:hashlib.sha256((directory.parent/name).read_bytes()).hexdigest()
                                 for name in ('audit_adiabatic_physical.py','adiabatic_sensitivity.py',
                                              'validate_rate.py','cold_sympy.py','thermal_sympy.py')},
                  cases=[])
    for state in selected:
        start = time.monotonic()
        result = evaluate(state, upstream['metadata']['B_MeV4'])
        result['wall_seconds'] = time.monotonic()-start
        report['cases'].append(result)
        (directory/'adiabatic-physical-audit.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(result), flush=True)
    # Compare only after every case has been integrated independently.
    with (directory/'adiabatic-speed-sensitivity.csv').open() as stream:
        published = list(csv.DictReader(stream))
    maximum_ratio_difference = 0.
    for result in report['cases']:
        reference = next(row for row in published
                         if float(row['T_0minus']) == result['T_0minus']
                         and float(row['target_a_0plus']) == result['target_a_0plus'])
        differences = [result['ratios'][i]-float(reference[f'ratio_{model}'])
                       for i, model in enumerate(('exact', 'eq23', 'sommerfeld_T2'))]
        result['ratio_difference_from_published'] = differences
        maximum_ratio_difference = max(maximum_ratio_difference, max(abs(d) for d in differences))
    spreads = {str(a):np.ptp(np.array([r['ratios'] for r in report['cases']
                                     if r['target_a_0plus'] == a]), axis=0).tolist()
               for a in (.9, .995)}
    report['maximum_absolute_ratio_difference_from_published'] = maximum_ratio_difference
    report['ratio_spread_across_upstream_temperatures'] = spreads
    report['passed'] = (maximum_ratio_difference < 1e-3
                        and max(max(values) for values in spreads.values()) < 1e-10)
    (directory/'adiabatic-physical-audit.json').write_text(json.dumps(report, indent=2)+'\n')
    assert report['passed'], report
    print('Physical-state audit passed.', flush=True)


if __name__ == '__main__':
    main()
