#!/usr/bin/env python3
"""Reproduce independent EOS/coordinate and numerical convergence checks.

Run from the repository root:
  ./bin/python3 ref-list/massless-weak-rate/adiabatic_verify.py

Needs NumPy and mpmath. The alternative coordinate uses an independently
implemented, 40-digit EOS inversion, but deliberately shares the validated
collision integral and diffusion prescription. This is not a second weak-rate
derivation or a front BVP. Existing production results are read only.
"""
from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import mpmath as mp
import numpy as np

import adiabatic_sensitivity as model


ANCHORS = (.146, .5, .9)
INDEPENDENT_COMPOSITIONS = (.005, .146, .5, .9, .995)
SPEED_RATIO_TOLERANCE = 1e-3
SOURCE_FILES = ('adiabatic_sensitivity.py', 'validate_rate.py',
                'cold_sympy.py', 'thermal_sympy.py')
# Exact pre-refresh driver version; its numerical recipes are unchanged below.
LEGACY_VERIFIER_SHA256 = '2bf1a970d429f2e343714aa8f05b619b621d0e268c9efd6046a3b383f470326c'


def source_hashes():
    return {name: hashlib.sha256((model.HERE/name).read_bytes()).hexdigest()
            for name in SOURCE_FILES}


def comparison(candidate, reference):
    candidate, reference = np.asarray(candidate), np.asarray(reference)
    ratios = np.sqrt(candidate/candidate[0])
    reference_ratios = np.sqrt(reference/reference[0])
    difference = ratios/reference_ratios-1
    return dict(integral_relative_difference=(candidate/reference-1).tolist(),
                speed_ratios=ratios.tolist(),
                reference_speed_ratios=reference_ratios.tolist(),
                speed_ratio_absolute_difference=(ratios-reference_ratios).tolist(),
                speed_ratio_relative_difference=difference.tolist(),
                max_abs_speed_ratio_relative_difference=float(np.max(np.abs(difference))),
                speed_ratios_pass=bool(np.all(np.abs(difference)<SPEED_RATIO_TOLERANCE)))


def cold_state_sweep():
    """Repeat the 3,025 states which exposed the cold inversion roundoff bug."""
    failures = []
    max_pressure_residual = 0.
    max_density_residual = 0.
    nodes = np.geomspace(1e-7, .1, 121)
    for a in model.COMPOSITIONS:
        trajectory = model.ColdInterfaceTrajectory(a)
        for z in nodes:
            try:
                state = trajectory.state(float(z))
                expected = trajectory.nB*np.array([1, 1+state['a'], 1-state['a']])
                density = model.number_density(state['chemical'], state['T'])
                max_density_residual = max(max_density_residual,
                                           float(np.max(np.abs(density/expected-1))))
                max_pressure_residual = max(max_pressure_residual,
                                            state['pressure_relative_residual'])
                if not (state['T']>0 and np.isfinite(state['minus_da_dz'])
                        and state['minus_da_dz']>=0):
                    raise ArithmeticError('Invalid temperature or Jacobian')
            except (ValueError, ArithmeticError) as error:
                failures.append(dict(a_0plus=a, z=float(z), error=str(error)))
    return dict(count=len(nodes)*len(model.COMPOSITIONS), z_min=float(nodes[0]),
                z_max=float(nodes[-1]), failures=failures,
                max_pressure_relative_residual=max_pressure_residual,
                max_density_relative_residual=max_density_residual,
                passed=not failures and max_pressure_residual<1e-12
                and max_density_residual<1e-12)


def independent_a_integral(a_0plus, outer_order=18, phase_order=16, tail=48):
    """Integrate in a=a_0plus*(1-y**6), solving T(a) independently.

    The cold end has T proportional to y**3. Thus da/dy proportional to y**5
    cancels D proportional to y**(-5), without using the production da/dz.
    High precision keeps 1-y**6 distinct from one at the smallest Gauss nodes.
    """
    with mp.workdps(40):
        a0, cold_mu = mp.mpf(a_0plus), mp.mpf(400)
        def chemical(a, t):
            def invert(factor):
                if t == 0:
                    return mp.root(factor, 3)
                p = mp.pi**2*t**2/3
                return 2*mp.sqrt(p)*mp.sinh(mp.asinh(factor/(2*p**mp.mpf('1.5')))/3)
            return tuple(invert(factor) for factor in (1, 1+a, 1-a))

        def pressure(a, t):
            return (sum(mu**4/(4*mp.pi**2)+mu**2*t**2/2
                        for mu in chemical(a, t))+19*mp.pi**2*t**4/36)

        cold_pressure = pressure(a0, mp.mpf(0))
        nB = float(cold_mu**3/mp.pi**2)
        nodes, weights = model.gauss(outer_order)
        total = np.zeros(3)
        max_residual = mp.mpf(0)
        for lo, hi in zip(model.OUTER_EDGES[:-1], model.OUTER_EDGES[1:]):
            half = (hi-lo)/2
            for y, weight in zip((lo+hi)/2+half*nodes, half*weights):
                a = a0*(1-mp.mpf(float(y))**6)
                lower, upper = mp.mpf(0), mp.mpf(1)
                for _ in range(100):
                    t = (lower+upper)/2
                    if pressure(a, t)>cold_pressure:
                        upper = t
                    else:
                        lower = t
                t = (lower+upper)/2
                max_residual = max(max_residual, abs(pressure(a, t)/cold_pressure-1))
                potentials = tuple(float(cold_mu*mu) for mu in chemical(a, t))
                temperature = float(cold_mu*t)
                total += (weight*6*a_0plus*y**5/nB
                          *model.diffusion(sum(potentials), temperature)
                          *model.three_rates(potentials, temperature, phase_order, tail))
        return dict(a_0plus=a_0plus, cold_mu=float(cold_mu),
                    coordinate='a=a_0plus*(1-y**6)',
                    eos_decimal_digits=40, eos_bisection_iterations=100,
                    outer_order=outer_order, phase_order=phase_order, tail=tail,
                    max_pressure_relative_residual=float(max_residual),
                    integrals=total.tolist())


def production_index(data):
    indexed = {}
    for key, value in data.items():
        identity = (value['a_0plus'], value['outer_order'], value['phase_order'], value['tail'])
        if identity in indexed and value['integrals']!=indexed[identity][1]['integrals']:
            raise ValueError(f'Conflicting production records for {identity}')
        indexed[identity] = (key, value)
    return indexed


def expected_production_records(compositions):
    records = []
    for a in compositions:
        settings = [(8, 12, 48), (12, 16, 48), (18, 16, 48)]
        if a in (*ANCHORS, .995):
            settings += [(18, 28, 48), (18, 28, 64)]
        records += [(a, outer, phase, tail) for outer, phase, tail in settings]
    return records


def load_complete_production(directory, wait_seconds):
    deadline = time.monotonic()+wait_seconds
    checkpoint = directory/'adiabatic-quadrature.json'
    provenance = directory/'adiabatic-quadrature-provenance.json'
    upstream_path = directory/'adiabatic-upstream.json'
    while True:
        data = json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
        upstream_bytes = upstream_path.read_bytes()
        upstream = json.loads(upstream_bytes)
        targets = {state['target_a_0plus'] for state in upstream['states']}
        saved_compositions = {value['a_0plus'] for value in data.values()}
        required = set(model.COMPOSITIONS)|targets|saved_compositions
        missing = sorted(set(expected_production_records(required))-set(production_index(data)))
        if not missing:
            metadata = json.loads(provenance.read_text())
            if metadata['source_sha256'] != source_hashes():
                raise RuntimeError('Production checkpoint source signature is stale')
            coverage = dict(required_compositions=sorted(required),
                            saved_composition_count=len(saved_compositions),
                            upstream_target_compositions=sorted(targets),
                            upstream_selected_state_count=len(upstream['states']),
                            upstream_sha256=hashlib.sha256(upstream_bytes).hexdigest(),
                            missing_refinements=[])
            return data, metadata, hashlib.sha256(checkpoint.read_bytes()).hexdigest(), coverage
        if time.monotonic()>=deadline:
            raise RuntimeError(f'Production checkpoint is incomplete ({len(missing)} missing keys)')
        print(f'Waiting for production checkpoint: {len(missing)} keys missing', flush=True)
        time.sleep(min(30., max(0., deadline-time.monotonic())))


def finest(data, a):
    candidates = [value for value in data.values() if value['a_0plus']==a]
    return max(candidates, key=lambda value: (value['outer_order'],
                                              value['phase_order'], value['tail']))


def saved_refinement_comparisons(data):
    result = []
    indexed = production_index(data)
    for a in sorted({value['a_0plus'] for value in data.values()}):
        settings = [('outer', (12, 16, 48), (18, 16, 48))]
        if a in (*ANCHORS, .995):
            settings += [('phase', (18, 16, 48), (18, 28, 48)),
                         ('tail', (18, 28, 48), (18, 28, 64))]
        for name, initial, refined in settings:
            initial_key, initial_result = indexed[(a, *initial)]
            refined_key, refined_result = indexed[(a, *refined)]
            result.append(dict(a_0plus=a, parameter=name,
                               initial_key=initial_key, refined_key=refined_key,
                               **comparison(initial_result['integrals'],
                                            refined_result['integrals'])))
    return result


def run_expensive_checks(result):
    result['cold_state_sweep'] = cold_state_sweep()
    print('Cold state sweep:', result['cold_state_sweep'], flush=True)
    result['independent_a_coordinate'] = []
    for a in INDEPENDENT_COMPOSITIONS:
        item = independent_a_integral(a)
        result['independent_a_coordinate'].append(item)
        print(f'Independent a-coordinate integral complete: a(0+)={a:g}', flush=True)

    result['eos_iteration_checks'] = []
    for a in ANCHORS:
        values = [model.integrated_estimates(model.ColdInterfaceTrajectory(a, root_iterations=n),
                                             outer_order=18, phase_order=16, tail=48)
                  for n in (64, 100)]
        result['eos_iteration_checks'].append(dict(a_0plus=a, initial_iterations=64,
                     refined_iterations=100, outer_order=18, phase_order=16, tail=48,
                     initial_integrals=values[0]['integrals'],
                     refined_integrals=values[1]['integrals'],
                     **comparison(values[0]['integrals'], values[1]['integrals'])))
        print(f'EOS iteration check complete: a(0+)={a:g}', flush=True)

    result['cutoff_checks'] = []
    for a in ANCHORS:
        trajectory = model.ColdInterfaceTrajectory(a)
        for fraction in (1e-3, 1e-5, 1e-7):
            item = model.cutoff_check(trajectory, fraction, phase_order=24, order=24)
            item.update(a_0plus=a, phase_order=24, log_temperature_order=24,
                        log_temperature_segments=8, tail=48,
                        leading_cold_tail=(model.cold_endpoint_integrand(trajectory)
                                           *fraction**(1/3)).tolist())
            result['cutoff_checks'].append(item)
            print(f'Cutoff check complete: a(0+)={a:g}, Tcut/Tinf={fraction:g}', flush=True)


def verification_recipe_signature():
    names = {'cold_state_sweep', 'independent_a_integral', 'run_expensive_checks'}
    tree = ast.parse(Path(__file__).read_text())
    definitions = [ast.dump(node, include_attributes=False) for node in tree.body
                   if isinstance(node, ast.FunctionDef) and node.name in names]
    settings = dict(anchors=ANCHORS, independent_compositions=INDEPENDENT_COMPOSITIONS,
                    sweep_compositions=model.COMPOSITIONS)
    return hashlib.sha256(json.dumps([definitions, settings], sort_keys=True).encode()).hexdigest()


def apply_production_comparisons(result, directory, wait_seconds):
    data, provenance, checkpoint_hash, coverage = load_complete_production(directory, wait_seconds)
    result['production_provenance'] = provenance
    result['production_checkpoint_sha256'] = checkpoint_hash
    result['production_record_count'] = len(data)
    result['production_coverage'] = coverage
    result['speed_ratio_relative_tolerance'] = SPEED_RATIO_TOLERANCE
    result['saved_refinement_checks'] = saved_refinement_comparisons(data)
    for item in result['eos_iteration_checks']:
        item.update(comparison(item['initial_integrals'], item['refined_integrals']))
    for item in result['independent_a_coordinate']:
        reference = finest(data, item['a_0plus'])
        item['reference_settings'] = {key:reference[key] for key in ('outer_order', 'phase_order', 'tail')}
        item['reference_integrals'] = reference['integrals']
        item.update(comparison(item['integrals'], reference['integrals']))
    for item in result['cutoff_checks']:
        reference = finest(data, item['a_0plus'])
        item['reference_integrals'] = reference['integrals']
        item['reference_settings'] = {key:reference[key] for key in ('outer_order', 'phase_order', 'tail')}
        item['raw_comparison'] = comparison(item['raw_integrals'], reference['integrals'])
        item['corrected_comparison'] = comparison(item['corrected_integrals'], reference['integrals'])

    checked = (result['independent_a_coordinate']+result['eos_iteration_checks']
               +result['saved_refinement_checks']
               +[item['corrected_comparison'] for item in result['cutoff_checks']])
    result['source_unchanged_during_verification'] = result['source_sha256']==source_hashes()
    result['max_abs_checked_speed_ratio_relative_difference'] = max(
        item['max_abs_speed_ratio_relative_difference'] for item in checked)
    result['max_abs_checked_speed_ratio_absolute_difference'] = max(
        max(abs(value) for value in item['speed_ratio_absolute_difference']) for item in checked)
    result['passed'] = (result['cold_state_sweep']['passed']
                        and result['source_unchanged_during_verification']
                        and all(item['speed_ratios_pass'] for item in checked))
    result['acceptance_note'] = ('The tolerance applies to independent, refined, and cold-tail-corrected '
                                'speed ratios. Raw cutoff integrals intentionally omit physical support '
                                'and are reported without requiring agreement.')


def write_summaries(directory, result):
    rows = []
    def add(category, item, parameter='', cutoff_fraction=''):
        row = dict(check=category, a_0plus=item.get('a_0plus', ''), parameter=parameter,
                   cutoff_fraction=cutoff_fraction,
                   required_for_acceptance=category!='cutoff_raw',
                   speed_ratios_pass=item['speed_ratios_pass'])
        for i, name in enumerate(model.MODEL_NAMES):
            row[f'I2_relative_difference_{name}'] = item['integral_relative_difference'][i]
            row[f'ratio_relative_difference_{name}'] = item['speed_ratio_relative_difference'][i]
        rows.append(row)
    for category, key in (('independent_a_coordinate', 'independent_a_coordinate'),
                           ('eos_iterations', 'eos_iteration_checks'),
                           ('saved_refinement', 'saved_refinement_checks')):
        for item in result[key]:
            add(category, item, item.get('parameter', ''))
    for item in result['cutoff_checks']:
        for kind in ('raw', 'corrected'):
            add(f'cutoff_{kind}', dict(a_0plus=item['a_0plus'], **item[f'{kind}_comparison']),
                cutoff_fraction=item['cutoff_fraction'])
    with (directory/'adiabatic-convergence.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def scientific(value):
        if value == 0:
            return r'\(0\)'
        mantissa, exponent = f'{value:.2e}'.split('e')
        return rf'\({mantissa}\times10^{{{int(exponent)}}}\)'
    groups = [('Independent coordinate', result['independent_a_coordinate']),
              ('EOS iterations 64 to 100', result['eos_iteration_checks'])]
    for parameter, label in (('outer', 'Outer order 12 to 18'),
                             ('phase', 'Phase order 16 to 28'), ('tail', 'Tail 48 to 64')):
        groups.append((label, [item for item in result['saved_refinement_checks']
                              if item['parameter']==parameter]))
    for fraction in (1e-3, 1e-5, 1e-7):
        for kind in ('raw', 'corrected'):
            label = (rf'Cutoff {scientific(fraction)}, '
                     + ('uncorrected' if kind=='raw' else 'cold tail added'))
            groups.append((label, [item[f'{kind}_comparison'] for item in result['cutoff_checks']
                                   if item['cutoff_fraction']==fraction]))
    coverage = result['production_coverage']
    cold = result['cold_state_sweep']
    lines = [r'% Generated by adiabatic_verify.py; do not edit by hand.',
             rf'Numerical verification covers {coverage["saved_composition_count"]} distinct interface compositions and {result["production_record_count"]} quadrature records.',
             r'Outer-order refinement covers every composition; phase-order and tail checks cover four representative compositions.',
             rf'The cold-state sweep evaluates {cold["count"]} states, with {len(cold["failures"])} failures and maximum relative pressure and density residuals {scientific(cold["max_pressure_relative_residual"])} and {scientific(cold["max_density_relative_residual"])}, respectively.',
             r'An independently implemented 40-digit EOS and composition-coordinate integral check the temperature-coordinate Jacobian; the rate and diffusion routines are shared.',
             r'\par', r'\begin{table}[htbp]', r'\centering', r'\small',
             r'\begin{tabular}{lrr}', r'\hline',
             r'Check & Integral change & Speed-ratio change \\',
             r'\hline']
    for label, items in groups:
        integral = max(max(abs(v) for v in item['integral_relative_difference']) for item in items)
        ratio = max(item['max_abs_speed_ratio_relative_difference'] for item in items)
        lines.append(f'{label} & {scientific(integral)} & {scientific(ratio)} '+r'\\')
    lines += [r'\hline', r'\end{tabular}',
              r'\caption{Observed relative numerical changes for the cold-interface adiabatic sensitivity calculation, taking the maximum absolute value over the tested compositions and rate models.',
              r'Cutoffs are specified by \(T_{\rm cut}/T(\infty)\); uncorrected rows intentionally omit the cold contribution and are not acceptance checks.',
              (r'Every independent, refined, and corrected speed ratio agrees within the required relative tolerance \(10^{-3}\).}'
               if result['passed'] else r'The acceptance checks have not all passed.}'),
              r'\label{tab:adiabatic-verification}', r'\end{table}', r'\par', '']
    (directory/'adiabatic-verification.tex').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=model.HERE/'results')
    parser.add_argument('--production-wait-seconds', type=float, default=0.)
    parser.add_argument('--refresh-production', action='store_true',
                        help='Reuse checked numerical recipes and refresh production coverage/summaries')
    args = parser.parse_args()
    started = time.monotonic()
    signature = verification_recipe_signature()
    verifier_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output = args.results/'adiabatic-independent-verification.json'
    if args.refresh_production:
        result = json.loads(output.read_text())
        if result['source_sha256'] != source_hashes():
            raise RuntimeError('Physics source changed; expensive independent checks must be rerun')
        if result.get('verification_recipe_sha256') != signature:
            if result.get('verifier_sha256') != LEGACY_VERIFIER_SHA256:
                raise RuntimeError('Verification recipe changed; expensive independent checks must be rerun')
            result['legacy_recipe_migrated_from_sha256'] = LEGACY_VERIFIER_SHA256
        result.setdefault('independent_checks_verifier_sha256', result['verifier_sha256'])
        result['refreshed_at_utc'] = datetime.now(timezone.utc).isoformat()
    else:
        result = dict(schema_version=2, generated_at_utc=datetime.now(timezone.utc).isoformat(),
                      model_names=list(model.MODEL_NAMES),
                      speed_ratio_relative_tolerance=SPEED_RATIO_TOLERANCE,
                      scope='Independent EOS and integration coordinate; shared rates and diffusion; no BVP',
                      source_sha256=source_hashes(),
                      independent_checks_verifier_sha256=verifier_hash,
                      numpy_version=np.__version__, mpmath_version=mp.__version__)
        run_expensive_checks(result)
        result['independent_checks_wall_seconds'] = time.monotonic()-started
    result.update(schema_version=2, verifier_sha256=verifier_hash,
                  verification_recipe_sha256=signature)
    apply_production_comparisons(result, args.results, args.production_wait_seconds)
    result['last_run_wall_seconds'] = time.monotonic()-started
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    write_summaries(args.results, result)
    print(json.dumps(dict(output=str(output), passed=result['passed'],
                          max_abs_checked_speed_ratio_relative_difference=
                          result['max_abs_checked_speed_ratio_relative_difference'],
                          wall_seconds=result['last_run_wall_seconds']), indent=2), flush=True)
    if not result['passed']:
        raise SystemExit('Independent verification failed; inspect the saved report')


if __name__ == '__main__':
    main()
