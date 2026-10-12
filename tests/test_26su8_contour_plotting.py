from __future__ import annotations

import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "26-07-19.ipynb"
LOG_DATA_MARKER = (
    'LOG_T_DATA_PATH = "new_paper_calculations/26-07-18/'
    'uN-two-block-adaptive-contour-log.npy"'
)
LINEAR_DATA_MARKER = (
    'LINEAR_T_DATA_PATH = "new_paper_calculations/26-07-18/'
    'uN-two-block-adaptive-contour-linear-v2.npy"'
)
LOG_PNG_MARKER = (
    'CLUSTER_CONTOUR_PNG_FILENAME = '
    '"analytic-two-block-contour-log-entropy-boundary.png"'
)
LINEAR_PNG_MARKER = (
    'CLUSTER_CONTOUR_PNG_FILENAME = '
    '"analytic-two-block-contour-linear-v2-entropy-boundary.png"'
)


class ContourNotebookStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        cls.code_cells = [
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        ]

    def _single_cell_with(self, marker):
        matching = [
            (index, source)
            for index, source in enumerate(self.code_cells)
            if marker in source
        ]
        self.assertEqual(len(matching), 1)
        return matching[0]

    def test_log_and_linear_figures_are_separate_cells(self):
        log_index, log_source = self._single_cell_with(LOG_DATA_MARKER)
        linear_index, linear_source = self._single_cell_with(LINEAR_DATA_MARKER)

        self.assertNotEqual(log_index, linear_index)
        self.assertNotIn(LINEAR_DATA_MARKER, log_source)
        self.assertNotIn(LOG_DATA_MARKER, linear_source)
        self.assertIn(LOG_PNG_MARKER, log_source)
        self.assertIn(LINEAR_PNG_MARKER, linear_source)

    def test_each_visualization_has_a_tuning_block(self):
        self.assertTrue(
            self._single_cell_with(LOG_DATA_MARKER)[1].startswith(
                "# Tunable plotting parameters\n"
            )
        )
        self.assertTrue(
            self._single_cell_with(LINEAR_DATA_MARKER)[1].startswith(
                "# Tunable plotting parameters\n"
            )
        )

    def test_log_transform_precedes_contours_and_inline_labels(self):
        _, source = self._single_cell_with(LOG_DATA_MARKER)
        scale_index = source.index("ax.set_yscale(CLUSTER_CONTOUR_Y_SCALE)")
        limits_index = source.index(
            "ax.set_ylim(*CLUSTER_CONTOUR_Y_LIMITS_MEV)"
        )
        contour_index = source.index("contour = ax.contour(")
        label_index = source.index("ax.clabel(")

        self.assertLess(scale_index, contour_index)
        self.assertLess(limits_index, contour_index)
        self.assertLess(contour_index, label_index)

    def test_entropy_shading_copies_the_shared_seam_row(self):
        for marker in (LOG_DATA_MARKER, LINEAR_DATA_MARKER):
            _, source = self._single_cell_with(marker)
            self.assertIn(
                "entropy_jump[0, :] = low_entropy_jump[-1, :]",
                source,
            )

    def test_log_panel_uses_requested_ranges_and_levels(self):
        _, source = self._single_cell_with(LOG_DATA_MARKER)
        self.assertIn('CLUSTER_CONTOUR_X_SCALE = "log"', source)
        self.assertIn("CLUSTER_CONTOUR_X_LIMITS = (1.0e-4, 0.9)", source)
        self.assertIn(
            "CLUSTER_CONTOUR_Y_LIMITS_MEV = (1.0e-2, 50.0)",
            source,
        )
        self.assertIn(
            "CLUSTER_CONTOUR_LEVELS_M_S = "
            "(1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0)",
            source,
        )
        self.assertLess(
            source.index("ax.set_xscale(CLUSTER_CONTOUR_X_SCALE)"),
            source.index("contour = ax.contour("),
        )


if __name__ == "__main__":
    unittest.main()
