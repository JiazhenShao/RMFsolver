#!/usr/bin/env python3
"""Redraw all seven manuscript figures from saved, verified numerical data."""
import csv
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR','/private/tmp/massless-mpl')
import matplotlib
matplotlib.use('Agg')

from validate_rate import HERE,read_grid
from rate_figures import make_plots as rate_plots
from speed_sensitivity import make_plot as isothermal_plot,with_eq23_muB
from adiabatic_report import load_results,make_plots as adiabatic_plots


def main():
    output=HERE/'results'
    rate_plots(output,read_grid(output/'rate-comparison-grid.csv'))
    with (output/'speed-sensitivity.csv').open() as stream:
        rows=[{key:float(value) for key,value in row.items()} for row in csv.DictReader(stream)]
    isothermal_plot(output,with_eq23_muB(output,rows))
    upstream,_,best,physical=load_results(output)
    adiabatic_plots(output,upstream,best,physical)
    print('Redrew all seven figures from saved results at the manuscript text width.')


if __name__=='__main__':
    main()
