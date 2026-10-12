from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_PATH = SCRIPT_DIR / "build_T0plus0p1_state_of_art_contour.py"


def load_module():
    spec = importlib.util.spec_from_file_location("build_state_of_art", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class StateOfArtPayloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.builder = load_module()
        cls.payload = cls.builder.build_payload()

    def test_contains_every_unique_successful_point(self):
        records = self.payload["all_successful_records"]
        keys = {
            (int(record["theta_index"]), round(float(record["radial_fraction_used"]), 10))
            for record in records
        }
        self.assertEqual(len(records), 614)
        self.assertEqual(len(keys), len(records))
        self.assertTrue(all(bool(record["success"]) for record in records))

    def test_ready_plot_grid_is_complete(self):
        self.assertEqual(self.payload["plot_grid_shape"], (30, 18))
        self.assertTrue(np.all(self.payload["plot_success_grid"]))
        self.assertTrue(np.all(np.isfinite(self.payload["plot_u_0minus_m_s_grid"])))

    def test_outer_shell_brackets_5000_everywhere(self):
        outer = np.asarray(self.payload["plot_u_0minus_m_s_grid"], dtype=float)[:, -1]
        self.assertGreater(float(np.min(outer)), 5000.0)
        self.assertTrue(self.payload["complete_5000_contour_bracket"])


if __name__ == "__main__":
    unittest.main()
