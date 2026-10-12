from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

import RMFsolver.RMFparameter as para
import RMFsolver.Solver as solver


class SolverLowTemperatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        solver._set_couplings(para.paraQMCRMF3)

    def test_low_temperature_rmf_sources_use_zero_temperature_branch(self):
        state = (1200.0, 0, 30.0, 20.0, -3.0)
        scalar_density_at_zero = solver._ns(0.0, *state)
        baryon_density_at_zero = solver._nB(0.0, *state)

        with patch.object(
            solver,
            "quad",
            side_effect=AssertionError("low-temperature branch called quad"),
        ):
            scalar_density_at_low_T = solver._ns(0.01, *state)
            baryon_density_at_low_T = solver._nB(0.01, *state)

        np.testing.assert_allclose(
            scalar_density_at_low_T,
            scalar_density_at_zero,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            baryon_density_at_low_T,
            baryon_density_at_zero,
            rtol=0.0,
            atol=0.0,
        )


if __name__ == "__main__":
    unittest.main()
