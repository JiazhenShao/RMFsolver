"""Killable fixed-density EOS calculation; imports numerical libraries only here."""
import os
import sys
import warnings


def preflight(import_paths=()):
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[name] = '1'
    sys.path[:0] = list(import_paths)
    import scipy
    from RMFsolver import phase_velocity  # Import all compiled dependencies before reserving states.
    return dict(scipy_version=scipy.__version__, scipy_path=scipy.__file__)


def evaluate(point, physics, import_paths=()):
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[name] = '1'
    sys.path[:0] = list(import_paths)
    import numpy as np
    import scipy
    from RMFsolver import phase_velocity as pv
    import isothermal_mfp as mfp
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        T = point['T_0minus']
        upstream = pv._validated_pnm_state_from_nB(point['nB_0minus'], T)
        P = upstream['P_0minus']
        muB_inf = pv._solve_muB_inf_at_muK0_for_given_Pi(
            P, 0., physics['B_one_forth'], T, ms=physics['ms'], upB=physics['upB'])
        endpoint = pv._quark_thermo_state(muB_inf, 0., physics['B_one_forth'], T, 0.,
                                        ms=physics['ms'], upB=physics['upB'])
        pressure_residual = abs(endpoint['P']-P)/max(abs(P),1.)
        if not np.isfinite(pressure_residual) or pressure_residual > 1.e-8:
            raise ValueError('Static pressure matching failed EOS residual validation')
        coefficients = pv._microphysics_from_quark_state_isothermal_baseline(muB_inf,T)
        full = mfp.transport_lengths(muB_inf,T,'full')
        landau = mfp.transport_lengths(muB_inf,T,'landau')
        if not (np.isclose(full['D_K'], coefficients['D'], rtol=1.e-12, atol=0.) and
                np.isclose(full['gamma_K'], coefficients['gamma'], rtol=1.e-12, atol=0.)):
            raise ValueError('Manuscript coefficients no longer match solver microphysics')
    return dict(success=True, validated=True, T_0minus=T, muB_inf=muB_inf,
                muK_inf=0., jB=0., upstream=upstream, endpoint=endpoint,
                pressure_relative_residual=pressure_residual,
                D_full=full['D_K'], D_landau=landau['D_K'], gamma_K=full['gamma_K'],
                warnings=list(dict.fromkeys(str(w.message) for w in caught)),
                runtime=dict(scipy_version=scipy.__version__, scipy_path=scipy.__file__))
