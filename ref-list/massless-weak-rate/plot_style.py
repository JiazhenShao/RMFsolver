"""Shared figure typography at the manuscript's actual 6.5-inch text width.

Figure PDFs are included at textwidth without further shrinking. All ordinary
plot text is at least 12 PDF points, above the 11-point manuscript body.
"""
from functools import wraps

FIGURE_WIDTH = 6.5
TEXT_SIZE = 12
TITLE_SIZE = 12.5
SUPTITLE_SIZE = 13
LINE_WIDTH = 1.7
MARKER_SIZE = 4
ENDPOINT_MARKER_SIZE = 5
DPI = 220
GRID_ALPHA = .2


def paper_style(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        import matplotlib as mpl
        with mpl.rc_context({'font.size':TEXT_SIZE,'axes.labelsize':TEXT_SIZE,
                             'axes.titlesize':TITLE_SIZE,'xtick.labelsize':TEXT_SIZE,
                             'ytick.labelsize':TEXT_SIZE,'legend.fontsize':TEXT_SIZE,
                             'figure.titlesize':SUPTITLE_SIZE,
                             'figure.labelsize':TEXT_SIZE}):
            return function(*args, **kwargs)
    return wrapped


def model_line(axis,x,y,style,**kwargs):
    _,label,color,linestyle,marker=style
    # Space markers on the displayed curve, independent of node clustering.
    markevery=kwargs.pop('markevery',.14)
    return axis.plot(x,y,label=label,color=color,linestyle=linestyle,marker=marker,
                     linewidth=LINE_WIDTH,markersize=MARKER_SIZE,
                     markevery=markevery,**kwargs)


def endpoint_label(axis,x,y,color='#0072B2',marker='s',offset=(-8,0)):
    axis.plot(x,y,marker=marker,color=color,markersize=ENDPOINT_MARKER_SIZE,
              linestyle='none',clip_on=False)
    axis.annotate(f'{y:.2f}',xy=(x,y),xytext=offset,textcoords='offset points',
                  ha='right',va='center',fontsize=TEXT_SIZE,color=color)


def model_legend(figure,styles):
    from matplotlib.lines import Line2D
    handles=[Line2D([],[],color=color,linestyle=line,marker=marker,
                    linewidth=LINE_WIDTH,markersize=MARKER_SIZE,label=label)
             for _,label,color,line,marker in styles]
    figure.legend(handles=handles,loc='lower center',ncol=1,fontsize=TEXT_SIZE,
                  frameon=False,handlelength=2.8,bbox_to_anchor=(.5,.006),labelspacing=.35)


def save_figure(figure,output,name):
    figure.savefig(output/f'{name}.pdf')
    figure.savefig(output/f'{name}.png',dpi=DPI)
