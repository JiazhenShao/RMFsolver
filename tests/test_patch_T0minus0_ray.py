import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT / "new_paper_calculations" / "26-08-12" / "patch_T0minus0_ray.py"
)
SPEC = importlib.util.spec_from_file_location("patch_T0minus0_ray", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class TestPatchT0minus0Ray(unittest.TestCase):
    def test_replaces_only_selected_point_records(self):
        original = [
            {"point_index": 0, "value": "old-zero"},
            {"point_index": 1, "value": "keep-one"},
            {"point_index": 2, "value": "old-two"},
        ]
        replacement = [
            {"point_index": 0, "value": "new-zero"},
            {"point_index": 2, "value": "new-two"},
        ]
        actual = MODULE._replace_point_records(
            original,
            replacement,
            {0, 2},
            sort_key=lambda row: row["point_index"],
        )
        self.assertEqual(
            actual,
            [
                {"point_index": 0, "value": "new-zero"},
                {"point_index": 1, "value": "keep-one"},
                {"point_index": 2, "value": "new-two"},
            ],
        )

    def test_patched_filename_preserves_extension(self):
        actual = MODULE._patched_path(
            Path("/tmp/output"),
            "T0plus5-full-analytic.npy",
            "-T0minus0-local-patched",
        )
        self.assertEqual(
            actual,
            Path(
                "/tmp/output/"
                "T0plus5-full-analytic-T0minus0-local-patched.npy"
            ),
        )

    def test_parser_accepts_local_numerical_budgets(self):
        args = MODULE._parser().parse_args(
            [
                "--point-timeout",
                "60",
                "--target-timeout",
                "120",
                "--max-bridge-depth",
                "0",
            ]
        )
        self.assertEqual(args.point_timeout, 60.0)
        self.assertEqual(args.target_timeout, 120.0)
        self.assertEqual(args.max_bridge_depth, 0)


if __name__ == "__main__":
    unittest.main()
