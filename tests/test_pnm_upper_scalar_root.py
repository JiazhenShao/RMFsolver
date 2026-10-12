"""Regression: the PNM validator must stay on the vacuum-connected scalar root.

The RMF scalar gap equation admits a third solution besides the physical root
and the already-guarded sigma < 0 root: a collapsed large-sigma root with a
much smaller effective mass.  It carries sigma > 0 and P > 0, so
_pnm_branch_rejection cannot see it, and inside narrow muB windows the first
branch seed converges onto it.  _validated_pnm_state returned the first seed
that passed, so nB(muB) grew a ~7.7x spike roughly 0.2 MeV wide.  Any bracketed
solve then converges onto the jump instead of the root, which is how
muB_from_nB_physical failed at nB = 9.1 n0, T = 0.01 MeV.
"""

import unittest
import warnings

import numpy as np

from RMFsolver import constants as const
from RMFsolver import phase_velocity as pv
from RMFsolver import RMFparameter as para


N0 = 0.16 * const.MeV_fm**3
TEMPERATURE_MEV = 1.0e-2
# The window where seed (30, 20, -3) lands on the collapsed root.
SPIKE_MUB_MEV = (2449.9, 2449.95, 2450.0)
FLANK_LOWER_MUB_MEV = 2449.8
FLANK_UPPER_MUB_MEV = 2450.1


class PNMUpperScalarRootTest(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore")

    def test_density_is_continuous_across_the_collapsed_root_window(self):
        lower = pv._validated_pnm_state(
            FLANK_LOWER_MUB_MEV, TEMPERATURE_MEV, para.paraQMCRMF3
        )[2]
        upper = pv._validated_pnm_state(
            FLANK_UPPER_MUB_MEV, TEMPERATURE_MEV, para.paraQMCRMF3
        )[2]
        span = upper - lower
        self.assertGreater(span, 0.0)
        for muB in SPIKE_MUB_MEV:
            with self.subTest(muB=muB):
                nB = pv._validated_pnm_state(
                    muB, TEMPERATURE_MEV, para.paraQMCRMF3
                )[2]
                # On the smooth branch nB is very nearly linear over 0.3 MeV.
                expected = lower + span * (
                    (muB - FLANK_LOWER_MUB_MEV)
                    / (FLANK_UPPER_MUB_MEV - FLANK_LOWER_MUB_MEV)
                )
                self.assertLess(
                    abs(nB - expected) / expected,
                    1.0e-6,
                    f"nB={nB:.6e} is off the vacuum-connected branch "
                    f"(expected about {expected:.6e})",
                )

    def test_selected_scalar_field_matches_the_flanking_branch(self):
        sigmas = [
            pv._validated_pnm_state(muB, TEMPERATURE_MEV, para.paraQMCRMF3)[3]
            for muB in (FLANK_LOWER_MUB_MEV,) + SPIKE_MUB_MEV
            + (FLANK_UPPER_MUB_MEV,)
        ]
        self.assertTrue(
            np.all(np.diff(sigmas) > 0.0),
            f"sigma is not monotone across the window: {sigmas}",
        )
        self.assertLess(max(sigmas) - min(sigmas), 1.0e-2)

    def test_analytic_nuclear_state_agrees_with_the_validator(self):
        # _analytic_nuclear_state screens the default-seed solve and only falls
        # back to _validated_pnm_state when the screen fires. The collapsed root
        # carries sigma > 0 and P > 0, so the screen passes and the upstream
        # state handed to analytic_velocity_bound is silently the wrong branch.
        for muB in (FLANK_LOWER_MUB_MEV,) + SPIKE_MUB_MEV + (FLANK_UPPER_MUB_MEV,):
            with self.subTest(muB=muB):
                state = pv._analytic_nuclear_state(
                    muB, TEMPERATURE_MEV, param=para.paraQMCRMF3, NM_type="PNM"
                )
                validated = pv._validated_pnm_state(
                    muB, TEMPERATURE_MEV, para.paraQMCRMF3
                )
                self.assertLess(
                    abs(state["nB_0minus"] - validated[2]) / validated[2], 1.0e-9
                )

    def test_inversion_recovers_muB_at_nine_saturation_densities(self):
        muB = pv.muB_from_nB_physical(
            9.1 * N0,
            TEMPERATURE_MEV,
            param=para.paraQMCRMF3,
            NM_type="PNM",
            muB_lo=2300.0,
            muB_hi=2600.0,
            scan_points=16,
            auto_expand=True,
        )
        achieved = pv._validated_pnm_state(
            muB, TEMPERATURE_MEV, para.paraQMCRMF3
        )[2]
        self.assertLess(abs(achieved - 9.1 * N0) / (9.1 * N0), 1.0e-6)


if __name__ == "__main__":
    unittest.main()
