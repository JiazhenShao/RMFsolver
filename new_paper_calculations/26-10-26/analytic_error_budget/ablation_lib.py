"""Read-only ablation: semi-analytic (full EOS) -> code's full-analytic, one ingredient at a time.

All runs use B^(1/4)=189.1565957288247 MeV, massless PNM, xi=-0.85. No files in the project are written.
"""
import json, sys, time, warnings
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import fsolve
warnings.simplefilter("ignore")
import RMFsolver.constants as const
import RMFsolver.phase_velocity as pv

N0 = 0.16 * const.MeV_fm**3
B = 189.1565957288247
U = B**4
XI = -0.85
STATES = [(8.860992751744538, 3.0268624251597), (27.83373842080244, 2.96875),
          (70.6735943750199, 1.434488038438507), (10.15113868209788, 4.500181704610503)]
x_gl, w_gl = leggauss(4000)
z_gl, wz = 0.5 * (x_gl + 1), 0.5 * w_gl


def speed(I2, lam, a):
    return np.sqrt(2 * I2 / (lam**2 * (1 - a) * (1 + XI * a)))


def closed_I2(muB, A, a):
    """Code/manuscript closed form (leading Landau D, cubic rate, ellipse at fixed muB)."""
    z = 1 - (a / A)**2
    bracket = 24 / 7 - 3 * z**(1 / 6) - 3 / 7 * z**(7 / 6)
    gam = pv._weak_rate_from_mu_q(muB / 3)["gamma"]
    return (9 * np.pi**(7 / 3) * gam / (8 * np.sqrt(2) * pv._TRANSPORT_H_CONST * pv._TRANSPORT_ALPHA_S**(5 / 3) * muB)
            * A**(7 / 3) * bracket)


def path_I2_variants(path, muB_ref):
    """I2 on the full-EOS path with (full D, local rate), (Landau D, local rate), (Landau D, cubic rate at muB_ref)."""
    nB = path.inf["nB"]
    w = path.coordinate(z_gl)
    muB, muK, T = path.fields(z_gl)
    a = path.thermo(w)[5]
    rate = path.rate(z_gl)
    qD = pv._TRANSPORT_QD_COEFF * muB / 3
    ratio_full_over_lead = 1 + np.pi**3 * np.cbrt(np.maximum(T, 0)) / (12 * pv._TRANSPORT_H_CONST * np.cbrt(qD))
    gam = pv._weak_rate_from_mu_q(muB_ref / 3)["gamma"]
    cubic = nB * gam * (a**3 + 81 * np.pi**2 * T**2 * a / muB_ref**2)
    dr = path.r_scale * wz
    return (np.sum(rate * dr) / nB**2, np.sum(rate * ratio_full_over_lead * dr) / nB**2,
            np.sum(cubic * ratio_full_over_lead * dr) / nB**2)


