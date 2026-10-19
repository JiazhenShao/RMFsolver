#!/usr/bin/env python3
"""Fresh massless weak-rate benchmarks for independent collaborator checks.

Run with Python 3, NumPy and SymPy. Uses the bundled, unchanged reference
modules. Every phase-space integral is recalculated; no saved rates are used
as inputs. Optional manuscript data are read only after calculation to check
agreement. The CSVs are numeric and retain double-precision values.
"""
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'reference'))
import numpy as np
import sympy
import validate_rate as rate
from cold_sympy import cold_phase_space
from thermal_sympy import phase_space_t2

TEMPERATURES = (0., .01, .1, 1., 5., 10., 20., 25., 50., 100.)
MU_B_VALUES = (900., 1200.)
COMPOSITIONS = (.146, .5, .9)
RELATIVE_TOLERANCE = 1e-6


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    started = time.monotonic()
    rows, checks = [], []
    for muB in MU_B_VALUES:
        for a in COMPOSITIONS:
            panel = []
            for temperature in TEMPERATURES:
                chemical = rate.neutral_fixed_a(muB, a, temperature)
                mu_u, mu_d, mu_s = chemical
                muK = mu_d-mu_s
                densities = rate.number_density(np.array(chemical), temperature)
                nB = float(sum(densities)/3)
                balance = -math.expm1(-muK/temperature) if temperature else 1.
                reverse_factor = math.exp(-muK/temperature) if temperature else 0.
                # Separate quadrature-order and energy-tail refinements.
                lower = rate.phase_space_forward(*chemical, temperature, n=24, tail=48)
                middle = rate.phase_space_forward(*chemical, temperature, n=40, tail=48)
                forward = rate.phase_space_forward(*chemical, temperature, n=40, tail=64)
                reverse = reverse_factor*forward
                net = balance*forward
                row = dict(mu_B_MeV=muB, a=a, T_MeV=temperature,
                           mu_u_MeV=mu_u, mu_d_MeV=mu_d, mu_s_MeV=mu_s,
                           mu_K_MeV=muK, n_B_MeV3=nB,
                           detailed_balance_factor=balance,
                           reverse_to_forward_factor=reverse_factor,
                           log_reverse_to_forward_factor=(-muK/temperature if temperature else None),
                           phase_space_forward_MeV8=forward,
                           phase_space_reverse_MeV8=reverse,
                           phase_space_net_MeV8=net,
                           weak_prefactor_MeV_minus4=rate.PREF,
                           rate_forward_MeV4=rate.PREF*forward,
                           rate_reverse_MeV4=rate.PREF*reverse,
                           rate_exact_net_MeV4=rate.PREF*net,
                           rate_Madsen_Eq23_MeV4=rate.eq23_rate(*chemical, temperature),
                           rate_Eq23_muB_over_3_MeV4=rate.eq23_rate(muB/3, mu_d, mu_s, temperature),
                           rate_derived_T0_T2_MeV4=rate.PREF*(cold_phase_space(*chemical)
                                                    +temperature**2*phase_space_t2(*chemical)))
                independent = (rate.finite_phase_space_3d(*chemical, temperature, n=24, tail=64)
                               if temperature in (20., 100.) else None)
                cold = cold_phase_space(*chemical) if temperature == 0 else None
                check = dict(mu_B_MeV=muB, a=a, T_MeV=temperature,
                             forward_n24_tail48_MeV8=lower,
                             forward_n40_tail48_MeV8=middle,
                             forward_n40_tail64_MeV8=forward,
                             order_relative_change=abs(middle/lower-1),
                             tail_relative_change=abs(forward/middle-1),
                             independent_3D_n24_tail64_forward_MeV8=independent,
                             independent_3D_relative_difference=(abs(independent/forward-1)
                                                                 if independent is not None else None),
                             analytic_cold_forward_MeV8=cold,
                             analytic_cold_relative_difference=(abs(forward/cold-1)
                                                               if cold is not None else None),
                             mu_B_relative_residual=abs(sum(chemical)/muB-1),
                             neutrality_residual_over_nB=abs(float(2*densities[0]-densities[1]-densities[2]))/nB,
                             composition_absolute_residual=abs(float((densities[1]-densities[2])/(2*nB))-a))
                assert all(math.isfinite(value) for value in row.values() if value is not None)
                rows.append(row)
                panel.append(row)
                checks.append(check)
            filename=f'phase-space_muB-{muB:g}_a-{a:g}.csv'
            write_csv(HERE/filename, panel)
            print(f'Calculated mu_B={muB:g} MeV, a={a:g}: {len(panel)} temperatures', flush=True)

    # Manuscript data are optional comparison evidence, never calculation inputs.
    saved_path = HERE.parents[2]/'ref-list/massless-weak-rate/results/rate-comparison-grid.csv'
    saved = {}
    if saved_path.exists():
        with saved_path.open() as stream:
            saved = {(float(r['muB']), float(r['a_cold']), float(r['T'])):float(r['exact_rate'])
                     for r in csv.DictReader(stream) if r['path']=='fixed_muB_a'}
    for row, check in zip(rows, checks):
        previous = saved.get((row['mu_B_MeV'], row['a'], row['T_MeV']))
        check['saved_manuscript_rate_MeV4'] = previous
        check['saved_manuscript_relative_difference'] = (abs(row['rate_exact_net_MeV4']/previous-1)
                                                          if previous is not None else None)
    metrics = ('order_relative_change', 'tail_relative_change', 'independent_3D_relative_difference',
               'analytic_cold_relative_difference', 'saved_manuscript_relative_difference',
               'mu_B_relative_residual', 'neutrality_residual_over_nB', 'composition_absolute_residual')
    maxima = {key:max((r[key] for r in checks if r[key] is not None), default=None) for key in metrics}
    metadata = dict(date='2026-10-15', timezone='America/Chicago',
                    generated_utc=datetime.now(timezone.utc).isoformat(),
                    reaction='d + u -> s + u; net = forward - reverse',
                    EOS_path='Fixed mu_B and neutral composition a at each temperature',
                    mu_B_MeV=MU_B_VALUES, a=COMPOSITIONS, temperatures_MeV=TEMPERATURES,
                    row_count=len(rows), exact_2D_order=40, exact_2D_tail=64,
                    convergence_relative_tolerance=RELATIVE_TOLERANCE,
                    constants=dict(G_F_MeV_minus2=rate.GF, theta_C_degrees=math.degrees(rate.THETA),
                                   weak_prefactor_MeV_minus4=rate.PREF),
                    versions=dict(python=platform.python_version(), numpy=np.__version__, sympy=sympy.__version__),
                    source_sha256={str(path.relative_to(HERE)):hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in [Path(__file__), *sorted((HERE/'reference').glob('*.py'))]},
                    validation_counts=dict(independent_3D=sum(r['independent_3D_relative_difference'] is not None for r in checks),
                                           analytic_cold=sum(r['analytic_cold_relative_difference'] is not None for r in checks),
                                           saved_manuscript=sum(r['saved_manuscript_relative_difference'] is not None for r in checks)),
                    maximum_residuals=maxima,
                    passed=all(value is None or value<RELATIVE_TOLERANCE for value in maxima.values()),
                    wall_seconds=time.monotonic()-started)
    write_csv(HERE/'phase-space_all.csv', rows)
    write_csv(HERE/'convergence.csv', checks)
    (HERE/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(metadata['maximum_residuals'], indent=2), flush=True)
    assert metadata['passed'], 'Validation failed; inspect convergence.csv before sharing.'
    print(f'Validated {len(rows)} states in {metadata["wall_seconds"]:.1f}s', flush=True)


if __name__ == '__main__':
    main()
