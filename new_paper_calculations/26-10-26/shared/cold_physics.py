"""Cold analytical calls; imports and recovery stay inside bounded children."""
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
METHODS = {
    'analytic': 'analytic_velocity_adiabatic_slow',
    'semi_analytic': 'semi_analytic_velocity_adiabatic_slow',
}
EXCLUSIONS = {
    'Quadratic isobar has no positive-temperature adiabatic equilibrium': 'stable_matter',
    'No positive-temperature equilibrated slow-front endpoint': 'stable_matter',
    'Quadratic interface fraction must satisfy 0 < a_0plus < 1': 'composition_ceiling',
    'Slow front requires an interior, K-rich interface': 'composition_ceiling',
    'Requested interface exceeds the physical composition branch': 'composition_ceiling',
}


def validate_result(result):
    """Infinite diffusion is required at this endpoint, not a failed finite check."""
    for key in ('velocity_m_s', 'jB', 'I2'):
        if not np.isfinite(result.get(key, np.nan)) or result[key] <= 0:
            raise ValueError('Nonfinite or nonpositive ' + key)
    if result.get('T_0plus') != 0 or not np.isposinf(result.get('D_K_0plus', np.nan)):
        raise ValueError('Exact cold endpoint was not preserved')
    if result.get('invD_K_0plus') != 0:
        raise ValueError('Cold inverse diffusion must be zero')
    if not 0 < result['a_0plus'] < 1:
        raise ValueError('Interface composition outside (0, 1)')
    errors = list(result['conservation_errors'].values())
    if not errors or not np.all(np.isfinite(errors)) or max(errors) > 1e-8:
        raise ValueError('Conservation check exceeds 1e-8')
    expected = dict(hydrodynamic_model='nonrelativistic_isobaric', closure='adiabatic',
                    alpha_s=.4, rate_approximation='madsen_muB_over_3')
    if any(result.get(k) != v for k, v in expected.items()):
        raise ValueError('Unexpected physical model metadata')


def evaluate(method, point, config):
    sys.path.insert(0, str(ROOT))
    from RMFsolver import phase_velocity as pv
    started = time.monotonic()
    function = getattr(pv, METHODS[method])
    options = dict(T_0plus=0., xi=config['xi'], ms=0., NM_type='PNM')
    original_seeds = pv._PNM_BRANCH_SEEDS
    extension = ()
    attempts = []
    try:
        for factor in ((1., .5, .2, .1) if method == 'semi_analytic' else (1.,)):
            if method == 'semi_analytic':
                options['path_rtol'] = config['path_rtol'] * factor
            for recovery in range(2):
                result = function(point['T_0minus_MeV'],
                    point['nB_0minus_over_n0'] * config['n0_MeV3'], config['B14_MeV'], **options)
                attempts.append(dict(path_rtol=options.get('path_rtol'),
                    seed_extension=extension, message=result.get('message'), success=result.get('success')))
                if (recovery == 0 and not extension and
                        'no two physical seeds agreed' in result.get('message', '')):
                    extension = ((35., 25., -3.), (40., 30., -3.))
                    pv._PNM_BRANCH_SEEDS = original_seeds + extension
                    continue
                break
            if result.get('message') != 'Slow-front EOS/diffusion table did not converge':
                break
    finally:
        pv._PNM_BRANCH_SEEDS = original_seeds
    status = 'accepted' if result.get('success') else (
        'excluded' if result.get('message') in EXCLUSIONS else 'failed')
    if status == 'accepted':
        validate_result(result)
    return dict(method=method, point=point, status=status,
        exclusion=EXCLUSIONS.get(result.get('message')), result=result,
        speed=result.get('velocity_m_s', np.nan), attempts=attempts,
        runtime_s=time.monotonic()-started)


def classify(method, point, config):
    """Cold domain from P and h/nB alone; no diffusion table or speed integral.

    On the full massless isobar, the physical interval is between the cold
    equilibrated and unit-composition enthalpies per baryon. Quadratic matching
    uses its second-order endpoint and A_adb. Both criteria are independent of xi.
    """
    sys.path.insert(0, str(ROOT))
    from RMFsolver import phase_velocity as pv
    from scipy.optimize import brentq
    nB = point['nB_0minus_over_n0'] * config['n0_MeV3']
    bag = config['B14_MeV']
    original = pv._PNM_BRANCH_SEEDS
    try:
        try:
            upstream = pv._isothermal_upstream_nuclear_state(point['T_0minus_MeV'],nB,pv.para.paraQMCRMF3,'PNM')
        except RuntimeError as exc:
            if 'no two physical seeds agreed' not in str(exc):
                raise
            pv._PNM_BRANCH_SEEDS = original + ((35.,25.,-3.),(40.,30.,-3.))
            upstream = pv._isothermal_upstream_nuclear_state(point['T_0minus_MeV'],nB,pv.para.paraQMCRMF3,'PNM')
    finally:
        pv._PNM_BRANCH_SEEDS = original
    pressure, H = upstream['P_0minus'], upstream['h_0minus']/nB
    cold = pv._slow_isobar_state(pressure,0.,0.,bag,0.)
    lower = H - cold['h']/cold['nB']
    if method == 'analytic':
        C = 108*np.pi**2*(pressure+bag**4)
        lower = H - C**.25
        if lower > 0:
            muB = H*brentq(lambda x:x**4-2*x**3+C/H**4,0.,C**.25/H,xtol=1e-15)
            upper = 2/9 - (H/muB-1)
        else:
            # The stable side has no positive-temperature endpoint. Its
            # composition ceiling cannot exclude a state before coexistence.
            upper = 2/9
    else:
        def isobar(k):
            return pv._slow_isobar_state(pressure,k,0.,bag,0.)
        hi = 50.
        while isobar(hi)['a'] < 1:
            hi *= 1.5
            if hi > 2000:
                raise RuntimeError('Cannot bracket cold unit-composition endpoint')
        unit = isobar(brentq(lambda k:isobar(k)['a']-1,0.,hi,xtol=1e-10))
        upper = unit['h']/unit['nB']-H
    exclusion = 'stable_matter' if lower <= 0 else ('composition_ceiling' if upper <= 0 else None)
    return dict(method=method, point=point, evaluation='thermodynamics_only',
        status='excluded' if exclusion else 'interior', exclusion=exclusion,
        margins=dict(stable_matter=float(lower), composition_ceiling=float(upper)),
        upstream=upstream)
