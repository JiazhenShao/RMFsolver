"""Publication-size local-rate figures from the saved phase-space grid.

Plotting is isolated here so figure styling cannot invalidate numerical
reference-source hashes in the conserved-trajectory checkpoints.
"""
from plot_style import (FIGURE_WIDTH,GRID_ALPHA,paper_style,model_line,
                        model_legend,save_figure)
from validate_rate import RATE_MODEL_STYLES,eq23_rate

# Local-rate figures only; the shared three-model styles stay unchanged.
EQ23_MUB_STYLE = ('eq23_muB_rate',r'Eq. (23), $\mu_u^5\to(\mu_B/3)^5$','#009E73','-.','D')
LOCAL_RATE_STYLES = RATE_MODEL_STYLES+(EQ23_MUB_STYLE,)
# One legend line taller than the three-model layout; panel sizes unchanged.
FIGURE_HEIGHT = 7.7


def eq23_muB_rate(mu_u,mu_d,mu_s,temperature):
    """Madsen Eq. (23) with mu_u**5 replaced by (mu_B/3)**5."""
    return eq23_rate((mu_u+mu_d+mu_s)/3,mu_d,mu_s,temperature)


@paper_style
def make_plots(output,rows):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator,FuncFormatter
    rows=[dict(r,eq23_muB_rate=eq23_muB_rate(r['mu_u'],r['mu_d'],r['mu_s'],r['T']))
          for r in rows]
    for muB in (900,1200):
        fig,axes=plt.subplots(3,2,figsize=(FIGURE_WIDTH,FIGURE_HEIGHT),sharex=True)
        for axis,a in zip(axes.flat,(.001,.146,.5,.9,.99,1)):
            selected=sorted((r for r in rows if r['path']=='fixed_muB_a'
                             and r['muB']==muB and r['a_cold']==a),key=lambda r:r['T'])
            for style in LOCAL_RATE_STYLES:
                model_line(axis,[r['T'] for r in selected],
                           [100*(r[style[0]]/r['exact_rate']-1) for r in selected],style)
            axis.set_xscale('log')
            axis.set_yscale('symlog',linthresh=1)
            axis.yaxis.set_major_formatter(FuncFormatter(lambda value,position:f'{value:g}%'))
            axis.xaxis.set_major_locator(FixedLocator([.01,.1,1,10,100]))
            axis.margins(y=.12)
            axis.set_title(rf'$a={a:g}$')
            axis.grid(alpha=GRID_ALPHA)
        fig.suptitle(rf'Fixed neutral composition, $\mu_B={muB}\,\mathrm{{MeV}}$')
        fig.supxlabel('Temperature [MeV]',y=.161)
        fig.supylabel('Relative rate error [%] (symlog)',x=.015)
        model_legend(fig,LOCAL_RATE_STYLES)
        fig.subplots_adjust(left=.18,right=.98,bottom=.247,top=.903,hspace=.42,wspace=.43)
        save_figure(fig,output,f'approximation-errors-muB-{muB}')
        plt.close(fig)

    fig,axes=plt.subplots(3,2,figsize=(FIGURE_WIDTH,FIGURE_HEIGHT),sharex=True)
    for row_axes,a in zip(axes,(.146,.5,.9)):
        for axis,muB in zip(row_axes,(900,1200)):
            selected=sorted((r for r in rows if r['path']=='fixed_muB_a'
                             and r['muB']==muB and r['a_cold']==a),key=lambda r:r['T'])
            for style in LOCAL_RATE_STYLES:
                model_line(axis,[r['T'] for r in selected],[r[style[0]] for r in selected],style)
            axis.set_xscale('log')
            axis.set_yscale('log')
            axis.set_title(rf'$\mu_B={muB}$ MeV, $a={a:g}$')
            axis.xaxis.set_major_locator(FixedLocator([.01,.1,1,10,100]))
            axis.grid(alpha=GRID_ALPHA)
    fig.supxlabel('Temperature [MeV]',y=.161)
    fig.supylabel(r'Net rate [MeV$^4$]',x=.015)
    model_legend(fig,LOCAL_RATE_STYLES)
    fig.subplots_adjust(left=.15,right=.98,bottom=.247,top=.932,hspace=.42,wspace=.31)
    save_figure(fig,output,'exact-rates')
    plt.close(fig)
