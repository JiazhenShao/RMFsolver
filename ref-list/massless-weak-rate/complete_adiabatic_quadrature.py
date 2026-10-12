#!/usr/bin/env python3
"""Refine extra compositions selected by the upstream density scan."""
import hashlib
import json
import time
from adiabatic_sensitivity import HERE,ColdInterfaceTrajectory,integrated_estimates,write_csv,MODEL_NAMES


def main():
    output=HERE/'results'
    checkpoint=output/'adiabatic-quadrature.json'
    data=json.loads(checkpoint.read_text())
    upstream=json.loads((output/'adiabatic-upstream.json').read_text())
    provenance=json.loads((output/'adiabatic-quadrature-provenance.json').read_text())
    for name,digest in provenance['source_sha256'].items():
        if hashlib.sha256((HERE/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Regenerate base quadrature after source changes')
    for a in sorted({state['target_a_0plus'] for state in upstream['states']}):
        trajectory=ColdInterfaceTrajectory(a)
        for outer,phase,tail in ((8,12,48),(12,16,48),(18,16,48)):
            key=f'{a:.17g}:{outer}:{phase}:{tail}'
            if any(r['a_0plus']==a and (r['outer_order'],r['phase_order'],r['tail'])==(outer,phase,tail)
                   for r in data.values()):
                continue
            start=time.monotonic()
            result=integrated_estimates(trajectory,outer,phase,tail)
            result.update(outer_order=outer,phase_order=phase,tail=tail,wall_seconds=time.monotonic()-start)
            data[key]=result
            temporary=checkpoint.with_suffix('.tmp')
            temporary.write_text(json.dumps(data,indent=2)+'\n')
            temporary.replace(checkpoint)
            print(f'Additional a={a:.10g}, outer={outer}, phase={phase}: {result["ratios"]}',flush=True)
    rows=[]
    for a in sorted({result['a_0plus'] for result in data.values()}):
        final=max((r for r in data.values() if r['a_0plus']==a),
                  key=lambda r:(r['outer_order'],r['phase_order'],r['tail']))
        row={key:final[key] for key in ('a_0plus','cold_mu','T_inf','nB','max_pressure_relative_residual')}
        for i,name in enumerate(MODEL_NAMES):
            row[f'I2_{name}']=final['integrals'][i]
            row[f'ratio_{name}']=final['ratios'][i]
            row[f'constant_D_ratio_{name}']=final['constant_D_ratios'][i]
        rows.append(row)
    write_csv(output/'adiabatic-universal.csv',rows)


if __name__=='__main__':
    main()
