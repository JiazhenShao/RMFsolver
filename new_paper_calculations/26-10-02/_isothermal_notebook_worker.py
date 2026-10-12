"""Stdlib-only entry point: configure threads before importing numerical libraries."""
import os
import sys


def call(method, physical=None, physics=None, import_paths=()):
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name]='1'
    sys.path[:0]=list(map(str,import_paths))
    import notebook_isothermal
    if method=='preflight':
        return notebook_isothermal.runtime_info()
    if method=='analytical':
        return notebook_isothermal.evaluate_point(physical,physics)
    raise ValueError('Unknown worker method')
