"""Regression: the A-ceiling domain error must classify as an outer-domain hit.

phase_velocity raises the ceiling error under several message spellings after
the `_Qstar -> _0plus` rename.  evaluate_A_state only recognised the
pre-rename `A_boundary must satisfy ...` text, so a genuine A=1 hit was
recorded as `numerical_failure`; the boundary machinery then saw no
success-to-outer-domain transition and aborted the whole scan.
"""

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "new_paper_calculations" / "26-08-07"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location(
    "ray_grid_A_ceiling_test", SCRIPT_DIR / "_analytic_ray_grid.py"
)
ray_grid = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ray_grid)


CEILING_MESSAGES = (
    "a_0plus must satisfy 0 < a_0plus < 1",
    "a_0plus must satisfy 0 < a_0plus <= A_boundary < 1",
    "a_0plus_LTE must satisfy 0 < a_0plus_LTE < 1",
    "A_boundary must satisfy 0 < A_boundary < 1",
    "Analytic velocity eigenvalue scan failed: a_0plus must satisfy 0 < a_0plus < 1",
)


class ACeilingClassificationTest(unittest.TestCase):
    def _state_for(self, message):
        with mock.patch.object(
            ray_grid, "muB_from_nB", return_value=1100.0
        ), mock.patch.object(
            ray_grid.phase_velocity,
            "analytic_velocity_bound",
            side_effect=RuntimeError(message),
        ):
            return ray_grid.evaluate_A_state(
                nB_over_n0=9.1,
                temperature_MeV=1.0e-2,
                n0=1.0,
                B14=189.156595729,
                xi=-0.5,
                ms=0.0,
                nm_type="PNM",
                upB=5000,
                interface_fraction_mode="fixed_T_0plus",
                velocity_closure="closed_form",
            )

    def test_every_ceiling_spelling_is_reported_as_A_forbidden(self):
        for message in CEILING_MESSAGES:
            with self.subTest(message=message):
                self.assertEqual(self._state_for(message)["status"], "A_forbidden")

    def test_A_forbidden_is_an_outer_domain_status(self):
        self.assertIn("A_forbidden", ray_grid.OUTER_FORBIDDEN_STATUSES)
        self.assertEqual(ray_grid._outer_boundary_kind("A_forbidden"), "A_equals_one")

    def test_unrelated_failures_stay_numerical(self):
        row = self._state_for("nB -> muB inversion did not converge")
        self.assertEqual(row["status"], "numerical_failure")

    def test_reachability_message_still_maps_to_its_own_status(self):
        row = self._state_for(
            "T_0plus lies below the resolved conserved-layer trajectory"
        )
        self.assertEqual(row["status"], "interface_temperature_unreachable")


if __name__ == "__main__":
    unittest.main()
