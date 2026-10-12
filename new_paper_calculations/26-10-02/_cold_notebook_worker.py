"""Importable notebook worker; thread limits precede all scientific imports."""
import os
import sys


def call(physical,physics,controls,continuation,jB_guess,bounds,import_paths=()):
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key]='1'
    sys.path[:0]=list(import_paths)
    from notebook_isothermal_cold import evaluate_point
    return evaluate_point(physical,physics,controls,continuation,jB_guess,bounds)
