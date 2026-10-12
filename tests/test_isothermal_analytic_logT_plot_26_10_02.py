"""Portable analytical plotting contracts using synthetic saved data only."""
import builtins
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PLOT_PATH = ROOT / "new_paper_calculations/26-10-02/isothermal_analytic_logT/_logT_plot.py"


def target(temperature, density=3.05, status="success", level=50.):
    return dict(kind="base", level=level, T_0minus=temperature, status=status,
                root=dict(T_0minus=temperature, nB_0minus_over_n0=density,
                          velocity_m_s=level, result={"validated": True}))


def fixtures():
    geometry = dict(inner_boundary=np.array([[3., 0.], [2.9, 20.], [1., 65.]]),
                    outer_boundary=np.array([[5.3, 0.], [5.2, 20.], [1., 94.674]]))
    temperatures = [.01, .1, 1., 20., 40., 80., 120.]
    payload = dict(levels=[50.], temperatures=temperatures,
                   targets={str(T): target(T) for T in temperatures},
                   boundaries={float(T).hex(): dict(T_0minus=T, lower=1., upper=5.3,
                                                    status="success") for T in temperatures if T <= 20.},
                   identity={"physics": {"xi": -.85}, "inputs": {"domain": {"sha256": "domain-hash"}}},
                   domain_fingerprint="domain-fingerprint", updated_unix_s=123456.75)
    return geometry, payload


class AnalyticalLogTPlotTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("analytic_logT_plot_test", PLOT_PATH)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.geometry, self.payload = fixtures()

    def tearDown(self):
        plt.close("all")

    def plot(self, **kwargs):
        return self.module.plot_contours(self.geometry, self.payload, **kwargs)

    @staticmethod
    def segments(fig):
        return [line.get_xydata() for line in fig.axes[0].lines
                if str(line.get_gid()).startswith("contour:")]

    def test_style_is_preserved_and_title_identifies_analytical_xi(self):
        fig = self.plot()
        ax = fig.axes[0]
        np.testing.assert_allclose(fig.get_size_inches(), [7.2, 7.2])
        self.assertEqual(ax.get_xlim(), (1., 5.5))
        self.assertEqual(ax.get_ylim(), (.01, 120.))
        self.assertEqual(ax.get_yscale(), "log")
        self.assertEqual(ax.get_box_aspect(), 1.)
        self.assertEqual(ax.xaxis.label.get_fontsize(), 18.)
        self.assertEqual(ax.title.get_fontsize(), 20.)
        self.assertIn("Analytical", ax.get_title())
        self.assertIn(r"\xi=-0.85", ax.get_title())
        self.assertNotIn("Numerical", ax.get_title())

    def test_only_analytical_roots_form_contours_over_entire_schedule(self):
        segments = self.segments(self.plot())
        self.assertEqual(len(segments), 1)
        np.testing.assert_array_equal(segments[0][:, 1], self.payload["temperatures"])
        np.testing.assert_array_equal(segments[0][:, 0], [3.05] * 7)

    def test_old_warm_curves_and_roots_cannot_contaminate_plot(self):
        self.geometry.update(levels_m_s=np.array([50.]),
                             curve_vertices_0=np.array([[4.9, 20.], [4.8, 45.], [4.7, 70.]]),
                             roots_0=np.array([[4.8, 45.]]), curve_solved_0=np.array([True]))
        fig = self.plot()
        for segment in self.segments(fig):
            np.testing.assert_array_equal(segment[:, 0], np.full(len(segment), 3.05))
        roots = [c for c in fig.axes[0].collections
                 if str(c.get_gid()).startswith("analytical-roots:")]
        self.assertEqual(len(roots), 1)
        self.assertEqual(len(roots[0].get_offsets()), 7)
        self.assertFalse(any("join" in text.get_text().lower() for text in fig.axes[0].texts))

    def test_missing_failed_pending_or_outside_target_always_breaks_line(self):
        for status in (None, "unresolved", "pending", "outside_window"):
            with self.subTest(status=status):
                self.geometry, self.payload = fixtures()
                if status is None:
                    del self.payload["targets"]["40.0"]
                else:
                    self.payload["targets"]["40.0"]["status"] = status
                fig = self.plot()
                self.assertEqual(len(self.segments(fig)), 2)
                for segment in self.segments(fig):
                    self.assertFalse(np.min(segment[:, 1]) < 40. < np.max(segment[:, 1]))
                self.assertEqual(fig.logT_diagnostics["missing_base_targets"], int(status is None))

    def test_only_validated_finite_correct_temperature_roots_are_accepted(self):
        corruptions = ({"result": {"validated": False}}, {"result": {}},
                       {"nB_0minus_over_n0": float("nan")}, {"velocity_m_s": -1.},
                       {"T_0minus": 41.})
        for corrupt in corruptions:
            with self.subTest(corrupt=corrupt):
                self.geometry, self.payload = fixtures()
                self.payload["targets"]["1.0"]["root"].update(corrupt)
                for segment in self.segments(self.plot()):
                    self.assertFalse(np.min(segment[:, 1]) < 1. < np.max(segment[:, 1]))

    def test_duplicate_schedule_evidence_is_ambiguous_and_breaks_line(self):
        self.payload["targets"]["duplicate"] = target(1., density=4.)
        for segment in self.segments(self.plot()):
            self.assertFalse(np.min(segment[:, 1]) < 1. < np.max(segment[:, 1]))

    def test_unscheduled_or_nonbase_records_are_not_new_contour_vertices(self):
        self.payload["targets"]["extra"] = target(30., density=4.8)
        self.payload["targets"]["refinement"] = dict(target(35., density=4.9), kind="refinement")
        segments = self.segments(self.plot())
        np.testing.assert_array_equal(segments[0][:, 1], self.payload["temperatures"])

    def test_full_gray_upper_region_is_present_without_computed_boundaries(self):
        self.payload["boundaries"] = {}
        fig = self.plot()
        patches = fig.axes[0].patches
        self.assertEqual(len(patches), 2)
        for patch in patches:
            np.testing.assert_allclose(patch.get_facecolor(), to_rgba("0.72"))
        upper = next(patch for patch in patches if patch.get_gid() == "excluded:outer")
        for x in (1.001, 3., 5.499):
            self.assertTrue(upper.get_path().contains_point((x, 110.)))
        self.assertFalse(upper.get_path().contains_point((1.1, 50.)))
        self.assertGreaterEqual(np.min(upper.get_xy()[:, 1]), .01)

    def test_successful_exact_boundaries_override_saved_geometry_without_top_gap(self):
        self.payload["boundaries"][float(1.).hex()].update(lower=2.8, upper=4.8)
        self.payload["boundaries"][float(.1).hex()].update(status="failure", upper=4.7)
        patches = self.plot().axes[0].patches
        outer = next(p for p in patches if p.get_gid() == "excluded:outer")
        inner = next(p for p in patches if p.get_gid() == "excluded:inner")
        self.assertTrue(np.any(np.all(outer.get_xy() == [4.8, 1.], axis=1)))
        self.assertTrue(np.any(np.all(inner.get_xy() == [2.8, 1.], axis=1)))
        self.assertFalse(np.any(np.all(outer.get_xy() == [4.7, .1], axis=1)))
        for x in (1.001, 3., 5.499):
            self.assertTrue(outer.get_path().contains_point((x, 110.)))

    def test_archive_is_pickle_free_boundary_only_and_preserves_validation(self):
        self.geometry["curve_vertices_0"] = np.array([[4.9, 40.], [4.8, 80.]])
        self.payload["targets"]["40.0"]["root"]["result"]["profile"] = np.arange(10)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plot.npz"
            self.module.export_plot_archive(self.geometry, self.payload, path)
            with np.load(path, allow_pickle=False) as archive:
                self.assertTrue(all(archive[key].dtype != object for key in archive.files))
                self.assertNotIn("curve_vertices_0", archive.files)
                metadata = json.loads(str(archive["metadata_json"]))
                self.assertEqual(metadata["model"], "analytical-isothermal")
                self.assertEqual(metadata["xi"], -.85)
                self.assertEqual(metadata["domain_fingerprint"], "domain-fingerprint")
                self.assertEqual(metadata["checkpoint_updated_unix_s"], 123456.75)
            geometry, payload = self.module.load_plot_archive(path)
            self.assertEqual(set(geometry), {"inner_boundary", "outer_boundary"})
            np.testing.assert_array_equal(geometry["outer_boundary"], self.geometry["outer_boundary"])
            self.assertEqual(payload["identity"], self.payload["identity"])
            self.assertTrue(payload["targets"]["40.0"]["root"]["result"]["validated"])
            self.assertNotIn("profile", payload["targets"]["40.0"]["root"]["result"])
            before = self.segments(self.plot())
            after = self.segments(self.module.plot_contours(geometry, payload))
            self.assertEqual(len(before), len(after))
            for a, b in zip(before, after):
                np.testing.assert_array_equal(a, b)

    def test_old_numerical_archive_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.npz"
            np.savez(path, metadata_json=np.asarray(json.dumps({"schema_version": 1})),
                     payload_json=np.asarray("{}"))
            with self.assertRaises(ValueError):
                self.module.load_plot_archive(path)

    def test_empty_checkpoint_produces_no_fabricated_contours(self):
        self.payload["targets"] = {}
        fig = self.plot()
        self.assertEqual(self.segments(fig), [])
        self.assertEqual(fig.logT_diagnostics["missing_base_targets"], 7)
        self.assertEqual(len(fig.axes[0].collections), 0)

    def test_default_schedule_covers_full_positive_log_temperature_window(self):
        self.assertAlmostEqual(self.module.TEMPERATURES[0], .01)
        self.assertAlmostEqual(self.module.TEMPERATURES[-1], 120.)
        self.assertTrue(np.all(np.diff(self.module.TEMPERATURES) > 0))

    def test_figure_exports_are_atomic_and_have_analytical_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            observed = []
            original_replace = self.module.os.replace

            def check_replace(source, destination):
                source, destination = Path(source), Path(destination)
                self.assertEqual(source.parent, destination.parent)
                self.assertGreater(source.stat().st_size, 100)
                self.assertNotEqual(source, destination)
                observed.append(destination.suffix)
                return original_replace(source, destination)

            with mock.patch.object(self.module.os, "replace", side_effect=check_replace):
                self.plot(output_dir=tmp)
            self.assertEqual(observed, [".png", ".pdf"])
            self.assertEqual(sorted(path.name for path in Path(tmp).iterdir()),
                             ["isothermal-analytic-logT.pdf", "isothermal-analytic-logT.png"])

    def test_import_and_plot_do_not_import_solver_or_eos(self):
        original_import = builtins.__import__

        def import_without_solver(name, *args, **kwargs):
            if name == "RMFsolver" or name.startswith("RMFsolver."):
                self.fail("Saved-data plotting attempted to load the EOS/solver")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=import_without_solver):
            spec = importlib.util.spec_from_file_location("analytic_logT_no_solver", PLOT_PATH)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.plot_contours(self.geometry, self.payload)


if __name__ == "__main__":
    unittest.main()
