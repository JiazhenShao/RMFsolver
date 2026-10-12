from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "26-07-23.ipynb"


def _function_source(source: str, function_name: str) -> str:
    start = source.index(f"def {function_name}")
    end = source.find("\n\ndef ", start + 5)
    return source[start:] if end < 0 else source[start:end]


class EllipticalBoundaryRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook = json.loads(NOTEBOOK_PATH.read_text())
        cell = next(
            cell
            for cell in notebook["cells"]
            if cell.get("id") == "elliptical-ray-boundary-functions"
        )
        cls.function_source = _function_source(
            "".join(cell["source"]),
            "er_solve_A_transition_on_path",
        )

    def _load_solver(self, evaluator):
        namespace = {
            "np": np,
            "pd": pd,
            "ER_ROUND_DIGITS": 12,
            "ER_A_BISECTION_STEPS": 10,
            "ER_A_ACCEPT_ONE_MINUS_A": 5.0e-3,
            "ER_A_ACCEPT_BRACKET_RELATIVE_WIDTH": 1.0e-5,
            "er_nearest_cached_state": lambda *_args: None,
            "er_evaluate_A_cartesian": evaluator,
            "er_to_internal_coordinates": lambda density, _temperature: (
                float(density),
                1.0,
            ),
        }
        exec(
            compile(
                self.function_source,
                "<er_solve_A_transition_on_path>",
                "exec",
            ),
            namespace,
        )
        return namespace["er_solve_A_transition_on_path"]

    @staticmethod
    def _row(coordinate, status):
        if status == "success":
            return {
                "status": status,
                "A_boundary": 1.0 - 2.0e-6,
                "endpoint_solution": (1200.0, 50.0),
                "nuclear_state": {"coordinate": float(coordinate)},
                "message": "",
            }
        return {
            "status": status,
            "A_boundary": np.nan,
            "endpoint_solution": None,
            "nuclear_state": {"coordinate": float(coordinate)},
            "message": "synthetic unresolved eigenvalue point",
        }

    def test_accepts_converged_one_sided_bracket(self):
        def evaluator(density, _temperature, **_kwargs):
            if density <= 0.4999999:
                status = "success"
            elif density >= 0.5000001:
                status = "A_forbidden"
            else:
                status = "numerical_failure"
            return self._row(density, status)

        solver = self._load_solver(evaluator)
        result = solver(
            np.asarray([0.0, 1.0]),
            lambda coordinate: (float(coordinate), 10.0),
            label="synthetic_thin_unresolved_band",
        )

        self.assertEqual(result["status"], "root")
        self.assertTrue(result["accepted_from_one_sided_bracket"])
        self.assertLessEqual(
            result["coordinate_bracket_relative_width"],
            1.0e-5,
        )
        self.assertLessEqual(
            result["one_minus_A_residual"],
            5.0e-3,
        )

    def test_rejects_wide_unresolved_band(self):
        def evaluator(density, _temperature, **_kwargs):
            if density <= 0.4:
                status = "success"
            elif density >= 0.6:
                status = "A_forbidden"
            else:
                status = "numerical_failure"
            return self._row(density, status)

        solver = self._load_solver(evaluator)
        with self.assertRaisesRegex(
            RuntimeError,
            "recovery did not shrink bracket",
        ):
            solver(
                np.asarray([0.0, 1.0]),
                lambda coordinate: (float(coordinate), 10.0),
                label="synthetic_wide_unresolved_band",
            )


class AnalyticalContourPayloadIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook = json.loads(NOTEBOOK_PATH.read_text())
        cell = next(
            cell
            for cell in notebook["cells"]
            if cell.get("id") == "fc1a509d"
        )
        cls.cell_source = "".join(cell["source"])

    def test_analytical_cell_loads_its_own_uniquely_named_payload(self):
        self.assertIn(
            "RAY_CONTOUR_ANALYTIC_PAYLOAD_PATH",
            self.cell_source,
        )
        self.assertIn(
            "ray_analytic_contour_payload = np.load",
            self.cell_source,
        )
        self.assertNotIn(
            'ray_cluster_payload["',
            self.cell_source,
        )

    def test_ud_critical_curve_uses_zero_strangeness_constraint(self):
        self.assertIn(
            "ray_ud_critical_density",
            self.cell_source,
        )
        self.assertIn(
            'ray_ud_species["n_s"]',
            self.cell_source,
        )
        self.assertIn(
            "ray_ud_muB_Q = ray_ud_muB_N - ray_ud_muK",
            self.cell_source,
        )
        self.assertIn(
            'label=r"$n_{B,\\mathrm{crit}}^{N\\leftrightarrow ud}(T)$"',
            self.cell_source,
        )


if __name__ == "__main__":
    unittest.main()
