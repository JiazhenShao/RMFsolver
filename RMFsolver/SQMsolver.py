"""Simple bag-model helper functions for quark matter."""

import numpy as np

__all__ = ["PQM_bag"]


def _P_f(mu, m):
    """Return the zero-temperature free-fermion pressure in MeV^4."""
    if mu <= m:
        return 0.0
    k_F = np.sqrt(mu**2 - m**2)
    return ((2 * k_F**3 - 3 * m**2 * k_F) * mu + 3 * m**4 * np.log((k_F + mu) / m)) / (24 * np.pi**2)


def PQM_bag(muB, muK, bag_constant, m_s=0.0):
    """Return bag-model quark pressure at T=0.

    Inputs:
    - muB: baryon chemical potential in MeV
    - muK: charge/isospin chemical potential in MeV
    - bag_constant: bag constant in MeV^4
    - m_s: strange-quark mass in MeV

    Output:
    - quark-matter pressure in MeV^4
    """
    mu_u = muB / 3 + muK / 2
    mu_d = muB / 3
    mu_s = muB / 3 - muK / 2
    return (mu_u**4 + mu_d**4) * 3 / (12 * np.pi**2) + 3 * _P_f(mu_s, m_s) - bag_constant
