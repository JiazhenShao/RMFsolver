#!/usr/bin/env python3
"""Join physical upstream states to scale-invariant adiabatic quadrature.

Run adiabatic_upstream.py and adiabatic_sensitivity.py first. This command
generates the figures, physical-state table and LaTeX results without rerunning
the upstream EOS or collision integrals. Cached quadrature uses the common
target composition; its difference from the solved composition is recorded.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from adiabatic_sensitivity import (HERE, MODEL_NAMES, ColdInterfaceTrajectory,
                                  cold_endpoint_integrand, pressure_kinetic, write_csv)
from validate_rate import RATE_MODEL_STYLES
from rate_figures import LOCAL_RATE_STYLES


def load_results(output):
    provenance=json.loads((output/'adiabatic-quadrature-provenance.json').read_text())
    for name,digest in provenance['source_sha256'].items():
        if hashlib.sha256((HERE/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Quadrature source changed; regenerate before reporting')
    verification=json.loads((output/'adiabatic-independent-verification.json').read_text())
    checkpoint=output/'adiabatic-quadrature.json'
    if (not verification['passed'] or verification['production_checkpoint_sha256']!=
            hashlib.sha256(checkpoint.read_bytes()).hexdigest()):
        raise ValueError('Current quadrature needs passing verification before reporting')
    upstream=json.loads((output/'adiabatic-upstream.json').read_text())
    for name,digest in upstream['metadata']['source_sha256'].items():
        if hashlib.sha256((HERE.parents[1]/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Upstream EOS source changed; regenerate before reporting')
    quadrature=json.loads(checkpoint.read_text())
    best={}
    for result in quadrature.values():
        a=result['a_0plus']
        rank=lambda r:(r['outer_order'],r['phase_order'],r['tail'])
        if a not in best or rank(result)>rank(best[a]):
            best[a]=result
    physical=[]
    for state in upstream['states']:
        target=state['target_a_0plus']
        if target not in best:
            raise ValueError(f'Missing quadrature for requested composition {target}')
        result=best[target]
        scale=(math.pi**2*state['nB'])**(1/3)/result['cold_mu']
        row={key:state[key] for key in ('T_0minus','nB_0minus_over_n0','muB_0minus',
                      'P_0minus','h_over_nB_0minus','nB','a_0plus','target_a_0plus',
                      'target_a_0plus_residual','interval_id')}
        actual=ColdInterfaceTrajectory(state['a_0plus'],result['cold_mu']*scale)
        row.update(T_inf=actual.T_inf,muB_0plus=sum(actual.cold_chemical),
                   muB_inf=actual.state(1.)['muB'],T_0plus=0.,
                   muK_0plus=actual.state(0.)['muK'],muK_inf=actual.state(1.)['muK'],
                   cold_pressure_relative_residual=actual.pressure/(state['P_0minus']+upstream['metadata']['B14_MeV']**4)-1,
                   quadrature_a_0plus=target)
        for i,flavor in enumerate(('u','d','s')):
            row[f'mu_{flavor}_0plus']=actual.cold_chemical[i]
            row[f'mu_{flavor}_inf']=actual.state(1.)['chemical'][i]
        scaled_target=ColdInterfaceTrajectory(target,result['cold_mu']*scale)
        row['quadrature_pressure_relative_residual']=scaled_target.pressure/(state['P_0minus']+upstream['metadata']['B14_MeV']**4)-1
        for i,name in enumerate(MODEL_NAMES):
            row[f'I2_{name}']=result['integrals'][i]*scale**4
            row[f'ratio_{name}']=result['ratios'][i]
            row[f'constant_D_ratio_{name}']=result['constant_D_ratios'][i]
        physical.append(row)
    return upstream,quadrature,best,physical


def make_plots(output,upstream,best,physical):
    os.environ.setdefault('MPLCONFIGDIR','/private/tmp/massless-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    from plot_style import (FIGURE_WIDTH,TEXT_SIZE,GRID_ALPHA,LINE_WIDTH,MARKER_SIZE,
                            paper_style,model_line,endpoint_label,model_legend,save_figure)

    @paper_style
    def draw():
        fig,axes=plt.subplots(3,2,figsize=(FIGURE_WIDTH,7.55),sharex=True,sharey=True)
        ymax=max(max(result['ratios']) for result in best.values())
        ymin=min(min(result['ratios']) for result in best.values())
        for axis,slice_ in zip(axes.flat,upstream['slices']):
            temperature=slice_['T_0minus']
            selected=[r for r in physical if r['T_0minus']==temperature]
            if selected:
                lowest=min(r['a_0plus'] for r in selected)
                if lowest>.1:
                    axis.axvspan(0,lowest,color='#eeeeee',zorder=0)
                    axis.text(lowest/2,.5,'Unavailable',transform=axis.get_xaxis_transform(),
                              ha='center',fontsize=TEXT_SIZE,color='#666666')
            for interval in slice_['intervals']:
                # Two upstream roots can give the same composition and ratio.
                by_composition={r['target_a_0plus']:r for r in selected
                                if r['interval_id']==interval['interval_id']}
                segment=[by_composition[a] for a in sorted(by_composition)]
                markers=([0,len(segment)//2,len(segment)-1] if segment and
                         segment[-1]['a_0plus']-segment[0]['a_0plus']<.2 else .14)
                for name,style in zip(MODEL_NAMES,RATE_MODEL_STYLES):
                    model_line(axis,[r['a_0plus'] for r in segment],
                               [r[f'ratio_{name}'] for r in segment],style,markevery=markers)
                if segment:
                    last=segment[-1]
                    endpoint_label(axis,last['a_0plus'],last['ratio_eq23'])
            if not selected:
                message=('EOS scan unresolved' if slice_['status']=='incomplete_eos_scan' else
                         'No admissible\ncold-interface trajectory')
                axis.text(.5,.5,message,transform=axis.transAxes,ha='center',va='center',fontsize=TEXT_SIZE)
            axis.set(title=rf'$T(0^-)={temperature:g}\,\mathrm{{MeV}}$',xlim=(0,1),
                     ylim=(min(.94,ymin*.96),ymax*1.06),xticks=(0,.25,.5,.75,1),yticks=(1,1.5,2))
            axis.grid(alpha=GRID_ALPHA)
        fig.supxlabel(r'Interface composition $a(0^+)$',y=.135)
        fig.supylabel('Speed estimate / exact estimate',x=.015)
        fig.suptitle('Adiabatic EOS trajectories\n'+r'$T(0^+)=0$, constant $P$ and $h/n_B$')
        model_legend(fig,RATE_MODEL_STYLES)
        fig.subplots_adjust(left=.15,right=.95,bottom=.23,top=.87,hspace=.45,wspace=.28)
        save_figure(fig,output,'adiabatic-speed-sensitivity')
        plt.close(fig)

        example=next(r for r in physical if r['T_0minus']==20 and r['target_a_0plus']==.9)
        result=best[.9]
        scale=(math.pi**2*example['nB'])**(1/3)/result['cold_mu']
        records=sorted(result['records'],key=lambda r:r['z'],reverse=True)
        a=np.array([r['a'] for r in records])
        z=np.array([r['z'] for r in records])
        temperatures=np.array([r['T']*scale for r in records])
        D=np.array([r['D_K']/scale for r in records])
        fig,axes=plt.subplots(2,2,figsize=(FIGURE_WIDTH,6.6))
        axes[0,0].plot(np.r_[0,a,.9],np.r_[example['T_inf'],temperatures,0],color='#555555',
                       marker='o',markersize=MARKER_SIZE,markevery=10,linewidth=LINE_WIDTH)
        axes[0,0].set(xlabel=r'$a$',ylabel=r'$T\ [\mathrm{MeV}]$')
        axes[0,1].semilogy(a,D,color='#555555',marker='o',markersize=MARKER_SIZE,
                           markevery=10,linewidth=LINE_WIDTH)
        axes[0,1].set(xlabel=r'$a$',ylabel=r'$D_K\ [\mathrm{MeV}^{-1}]$')
        axes[0,1].text(.06,.94,r'$D_K\to\infty$'+'\nat the cold interface',ha='left',va='top',
                       transform=axes[0,1].transAxes,fontsize=TEXT_SIZE)
        cold_trajectory=ColdInterfaceTrajectory(.9)
        endpoint=dict(zip(MODEL_NAMES,cold_endpoint_integrand(cold_trajectory)))
        weighted={name:np.array([r[f'weighted_{name}'] for r in records]) for name in MODEL_NAMES}
        # Plotting-only (mu_B/3)^5 model: same state, so only the mu_u^5 factor changes.
        mu_factor=lambda mu:(sum(mu)/3/mu[0])**5
        weighted['eq23_muB']=weighted['eq23']*np.array([mu_factor((r['mu_u'],r['mu_d'],r['mu_s']))
                                                         for r in records])
        endpoint['eq23_muB']=endpoint['eq23']*mu_factor(cold_trajectory.cold_chemical)
        exact=np.r_[endpoint['exact'],weighted['exact'][::-1]]
        for name,style in zip(MODEL_NAMES+('eq23_muB',),LOCAL_RATE_STYLES):
            # D_K and the Jacobian are common, so this is the local rate error on the path.
            model_line(axes[1,0],np.r_[0,z[::-1]],
                       100*(np.r_[endpoint[name],weighted[name][::-1]]/exact-1),style)
            mass=np.array([r['quadrature_weight'] for r in records])*weighted[name]
            cumulative=(np.cumsum(mass)-mass/2)/sum(mass)
            model_line(axes[1,1],np.r_[0,a,.9],np.r_[0,cumulative,1],style)
        axes[1,0].set_yscale('symlog',linthresh=1)
        axes[1,0].yaxis.set_major_formatter(FuncFormatter(lambda value,position:f'{value:g}%'))
        axes[1,0].set(xlabel=r'$z=[T/T(\infty)]^{1/3}$',ylabel='Integrand vs. exact [%]',
                      ylim=(-.3,400))
        axes[1,1].set(xlabel=r'$a$',ylabel=r'Cumulative fraction of $I_2$',ylim=(0,1.02))
        for axis in axes.flat:
            axis.grid(alpha=GRID_ALPHA)
        fig.suptitle(r'$T(0^-)=20\,\mathrm{MeV}$, $a(0^+)=0.9$'+'\n'+
                     rf'$n_B(0^-)/n_0={example["nB_0minus_over_n0"]:.3f}$')
        model_legend(fig,LOCAL_RATE_STYLES)
        fig.tight_layout(rect=(0,.17,1,.94),h_pad=1.2,w_pad=1.0)
        save_figure(fig,output,'adiabatic-trajectory-diagnostics')
        plt.close(fig)

        fig,axes=plt.subplots(1,2,figsize=(FIGURE_WIDTH,4.1),sharex=True,sharey=True)
        ordered=sorted(best)
        for axis,key,title in zip(axes,('ratios','constant_D_ratios'),
                                  ('Local diffusion','Constant diffusion')):
            for i,style in enumerate(RATE_MODEL_STYLES):
                model_line(axis,ordered,[best[a][key][i] for a in ordered],style)
            endpoint_label(axis,ordered[-1],best[ordered[-1]][key][1])
            axis.set(title=title,xlabel=r'$a(0^+)$',xlim=(0,1))
            axis.grid(alpha=GRID_ALPHA)
        fig.supylabel('Speed estimate / exact estimate',x=.015)
        model_legend(fig,RATE_MODEL_STYLES)
        fig.tight_layout(rect=(.035,.23,1,1),w_pad=1.2)
        save_figure(fig,output,'adiabatic-diffusion-weighting')
        plt.close(fig)
    draw()


def write_tex(output,upstream,best,physical):
    lines=[r'\begin{table}[tbp]',r'\centering',
           r'\caption{Adiabatic cold-interface benchmarks on the conserved EOS trajectory.',
           r'The last two columns give percentage changes of the speed estimate relative to the exact phase-space reference.}',
           r'\label{tab:adiabatic-speed}',r'\begin{tabular}{rrrrrr}',r'\toprule',
           r'\(T(0^-)\) & \(a(0^+)\) & \(n_B(0^-)/n_0\) & \(T(\infty)\) & Eq.~(23) & \(T^0+T^2\)\\',
           r'{[MeV]} & & & {[MeV]} & {[\%]} & {[\%]}\\',r'\midrule']
    for slice_ in upstream['slices']:
        for a in (.146,.5,.9):
            selected=[r for r in physical if r['T_0minus']==slice_['T_0minus'] and r['target_a_0plus']==a]
            if not selected:
                lines.append(f"{slice_['T_0minus']:g} & {a:g} & "+r'\multicolumn{4}{c}{unavailable in density search range}\\')
            for row in selected:
                lines.append(f"{row['T_0minus']:g} & {row['target_a_0plus']:g} & {row['nB_0minus_over_n0']:.5f} & "
                             f"{row['T_inf']:.3f} & {100*(row['ratio_eq23']-1):+.2f} & {100*(row['ratio_sommerfeld_T2']-1):+.2f} "+r'\\')
    lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}'])
    for slice_ in upstream['slices']:
        if not any(row['T_0minus']==slice_['T_0minus'] for row in physical):
            conclusion=('the EOS scan remains unresolved' if slice_['status']=='incomplete_eos_scan' else
                        'the density scan finds no admissible cold-interface trajectory')
            lines.append(r'At \(T(0^-)='+f"{slice_['T_0minus']:g}"+r'\,\mathrm{MeV}\), '+conclusion+r'; no rate-ratio curve is extrapolated into that slice.')
        if any(a in (.146,.5,.9) for a in slice_['unreachable_targets']):
            lines.append(r'The unavailable benchmark compositions at \(T(0^-)='+f"{slice_['T_0minus']:g}"+
                         r'\,\mathrm{MeV}\) are '+', '.join(r'\('+f'{a:g}'+r'\)' for a in slice_['unreachable_targets'] if a in (.146,.5,.9))+'.')
    lines.extend([r'\begin{table}[tbp]',r'\centering',
        r'\caption{Admissible upstream density intervals in the search range \(1\le n_B(0^-)/n_0\le10\).',
        r'Boundary compositions \(0\) and \(1\) are limiting states, excluded from the rate-integral calculation.}',
        r'\label{tab:adiabatic-intervals}',r'\begin{tabular}{rccc}',r'\toprule',
        r'\(T(0^-)\) [MeV] & \(n_B(0^-)/n_0\) interval & limiting \(a(0^+)\) range & status\\',r'\midrule'])
    for slice_ in upstream['slices']:
        intervals=slice_['intervals']
        if not intervals:
            status='unresolved' if slice_['status']=='incomplete_eos_scan' else 'unavailable'
            lines.append(f"{slice_['T_0minus']:g} & --- & --- & {status} "+r'\\')
        for interval in intervals:
            lo,hi=interval['nB_0minus_over_n0_bounds']
            alo,ahi=interval.get('reachable_a_0plus_bounds',sorted(interval['a_0plus_at_density_bounds']))
            left='[' if 0<interval['a_0plus_at_density_bounds'][0]<1 and lo==1 else '('
            aleft='[' if alo>1e-6 else '('
            lines.append(f"{slice_['T_0minus']:g} & {left}{lo:.6f}, {hi:.6f}) & {aleft}{alo:.4f}, {ahi:.4f}) & admissible "+r'\\')
    lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}'])
    low,mid,high=(best[a] for a in (.146,.5,.9))
    lines.append(r'At \(a(0^+)=0.146,0.5,0.9\), Eq.~(23) gives speed-estimate changes of '
                 +', '.join(r'\('+f'{100*(r["ratios"][1]-1):+.2f}'+r'\%\)' for r in (low,mid,high))+r', respectively.')
    lines.append(r'At the same compositions the derived \(T^0+T^2\) prescription changes the estimate by '
                 +', '.join(r'\('+f'{100*(r["ratios"][2]-1):+.2f}'+r'\%\)' for r in (low,mid,high))+'.')
    lines.append(r'For \(a(0^+)=0.9\), replacing local diffusion by a common constant on the same thermodynamic trajectory changes the Eq.~(23) ratio from '
                 +r'\('+f'{high["ratios"][1]:.5f}'+r'\) to \('+f'{high["constant_D_ratios"][1]:.5f}'+r'\).')
    (HERE/'adiabatic-results.tex').write_text('\n'.join(lines)+'\n')


def save_physical_trajectories(output,upstream,best,physical):
    rows=[]
    max_pressure=max_neutrality=max_density=max_enthalpy=0.
    for physical_state in physical:
        target=physical_state['target_a_0plus']
        result=best[target]
        scale=(math.pi**2*physical_state['nB'])**(1/3)/result['cold_mu']
        for sample in result['records']:
            chemical=np.array([sample[f'mu_{f}']*scale for f in ('u','d','s')])
            temperature=sample['T']*scale
            density=chemical**3/math.pi**2+chemical*temperature**2
            pressure=pressure_kinetic(chemical,temperature)-upstream['metadata']['B14_MeV']**4
            pres=abs((pressure-physical_state['P_0minus'])/(physical_state['P_0minus']+upstream['metadata']['B14_MeV']**4))
            neutral=abs(2*density[0]-density[1]-density[2])/physical_state['nB']
            max_pressure=max(max_pressure,pres)
            max_neutrality=max(max_neutrality,neutral)
            max_density=max(max_density,abs(sum(density)/(3*physical_state['nB'])-1))
            enthalpy=4*(pressure+upstream['metadata']['B14_MeV']**4)/physical_state['nB']
            max_enthalpy=max(max_enthalpy,abs(enthalpy/physical_state['h_over_nB_0minus']-1))
            row={k:physical_state[k] for k in ('T_0minus','nB_0minus_over_n0','target_a_0plus','a_0plus','nB','interval_id')}
            row.update(a=sample['a'],z=sample['z'],T=temperature,muB=sample['muB']*scale,
                       muK=sample['muK']*scale,D_K=sample['D_K']/scale,P=pressure,
                       h_over_nB=enthalpy,
                       mu_u=chemical[0],mu_d=chemical[1],mu_s=chemical[2],
                       quadrature_weight=sample['quadrature_weight'],
                       abs_da_dz=sample['weighted_exact']*result['nB']/(sample['D_K']*sample['rate_exact']))
            for name in MODEL_NAMES:
                row[f'rate_{name}']=sample[f'rate_{name}']*scale**8
                row[f'weighted_{name}']=sample[f'weighted_{name}']*scale**4
            rows.append(row)
    write_csv(output/'adiabatic-trajectories.csv',rows)
    checks=dict(max_pressure_scaled_residual=max_pressure,max_neutrality_scaled_residual=max_neutrality,
                max_baryon_density_relative_residual=max_density,max_enthalpy_per_baryon_relative_residual=max_enthalpy,
                max_endpoint_muK_inf=max(abs(r['muK_inf']) for r in physical),
                max_cold_interface_temperature=max(abs(r['T_0plus']) for r in physical),
                trajectory_sample_count=len(rows))
    if max(max_pressure,max_neutrality,max_density,max_enthalpy)>1e-9:
        raise ArithmeticError(f'Physical trajectory failed conservation checks: {checks}')
    return checks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=HERE/'results')
    args=parser.parse_args()
    upstream,quadrature,best,physical=load_results(args.output)
    if len(upstream['slices'])!=6:
        raise ValueError('Upstream calculation is incomplete: expected six temperature slices')
    if any(s['status']=='incomplete_eos_scan' for s in upstream['slices']):
        raise ValueError('Resolve upstream EOS failures before producing final report')
    write_csv(args.output/'adiabatic-speed-sensitivity.csv',physical)
    trajectory_checks=save_physical_trajectories(args.output,upstream,best,physical)
    make_plots(args.output,upstream,best,physical)
    write_tex(args.output,upstream,best,physical)
    summary=dict(interpretation='Rate-integral speed estimates on a shared conserved EOS trajectory; no BVP',
        normalization='sqrt(I2_model/I2_exact)',T_0plus=0.,alpha_s=.3,
        units=dict(temperature='MeV',chemical_potential='MeV',pressure='MeV^4',density='MeV^3',
                   rate='MeV^4',diffusion='MeV^-1',I2='dimensionless',weighted_z_integrand='dimensionless'),
        scaling='At fixed a_0plus, T and mu scale as nB^(1/3), I2 as nB^(4/3); ratios are invariant',
        quadrature_composition='Common target grid; actual upstream composition residual is saved per row',
        max_target_composition_residual=max(abs(r['target_a_0plus_residual']) for r in physical),
        max_cold_pressure_relative_residual=max(abs(r['cold_pressure_relative_residual']) for r in physical),
        trajectory_checks=trajectory_checks,
        upstream_metadata=upstream['metadata'],slices=upstream['slices'],
        source_sha256={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in
                       ('adiabatic_sensitivity.py','adiabatic_upstream.py','adiabatic_report.py',
                        'complete_adiabatic_quadrature.py','validate_rate.py','cold_sympy.py','thermal_sympy.py')})
    (args.output/'adiabatic-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(f'Generated adiabatic report for {len(physical)} physical states.')


if __name__=='__main__':
    main()
