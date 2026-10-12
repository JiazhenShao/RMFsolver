"""Saved-data-only plotting tests: no EOS imports or numerical solves."""
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
PLOT_PATH = ROOT / "new_paper_calculations/26-10-01/isothermal_numeric_logT/_logT_plot.py"


def target(T, density=3.05, status="success", kind="base", level=50.):
    return dict(level=level, T_0minus=T, kind=kind, status=status,
                root=dict(nB_0minus_over_n0=density, T_0minus=T, velocity_m_s=level))


def fixtures():
    warm = dict(levels_m_s=np.array([50.]),
                inner_boundary=np.array([[3., 0.], [2.9, 20.], [1., 65.]]),
                outer_boundary=np.array([[5.3, 0.], [5.2, 20.], [1., 95.]]),
                curve_vertices_0=np.array([[3., 0.], [3.05, 20.], [3.1, 40.], [1., 70.]]),
                curve_solved_0=np.array([False, True, True]),
                roots_0=np.array([[3., 0.], [3.1, 40.]]))
    temperatures = [.01, 1., 20.]
    payload = dict(levels=[50.], temperatures=temperatures,
                   targets={str(T): target(T) for T in temperatures},
                   boundaries={float(T).hex(): dict(T_0minus=T, lower=3., upper=5.3,
                                                    status="success") for T in temperatures})
    return warm, payload


class LogTPlotTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(PLOT_PATH.is_file(), "The saved-data logT plotting module is not implemented")
        spec = importlib.util.spec_from_file_location("logT_plot_test", PLOT_PATH)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.warm, self.payload = fixtures()

    def tearDown(self):
        plt.close("all")

    def plot(self):
        return self.module.plot_contours(self.warm, self.payload)

    @staticmethod
    def segments(fig):
        return [line.get_xydata() for line in fig.axes[0].lines
                if str(line.get_gid()).startswith("contour:")]

    def test_log_window_fonts_and_full_gray_upper_region(self):
        fig = self.plot()
        ax = fig.axes[0]
        np.testing.assert_allclose(fig.get_size_inches(), [7.2, 7.2])
        self.assertEqual(ax.get_xlim(), (1., 5.5))
        self.assertEqual(ax.get_ylim(), (.01, 120.))
        self.assertEqual(ax.get_yscale(), "log")
        self.assertEqual(ax.get_box_aspect(), 1.)
        self.assertEqual(ax.xaxis.label.get_fontsize(), 18.)
        self.assertEqual(ax.title.get_fontsize(), 20.)
        upper = [patch for patch in ax.patches if patch.get_gid() == "excluded:outer:warm"][0]
        np.testing.assert_allclose(upper.get_facecolor(), to_rgba("0.72"))
        for x in [1.05, 3., 5.45]:
            self.assertTrue(upper.get_path().contains_point((x, 110.)))

    def test_outer_region_label_stays_inside_log_axis(self):
        fig = self.plot()
        fig.canvas.draw()
        ax = fig.axes[0]
        label = next(text for text in ax.texts if "geq" in text.get_text())
        self.assertLessEqual(label.get_window_extent().y1, ax.get_window_extent().y1)

    def test_missing_base_temperature_is_never_bridged(self):
        del self.payload["targets"]["1.0"]
        fig = self.plot()
        for segment in self.segments(fig):
            self.assertFalse(np.min(segment[:, 1]) < 1. < np.max(segment[:, 1]))
        self.assertEqual(fig.logT_diagnostics["missing_base_targets"], 1)

    def test_failed_and_pending_scheduled_refinements_are_gaps(self):
        for status in ("unresolved", "pending"):
            with self.subTest(status=status):
                self.payload["targets"]["refinement"] = target(.1, status=status, kind="refinement")
                fig = self.plot()
                for segment in self.segments(fig):
                    self.assertFalse(np.min(segment[:, 1]) < .1 < np.max(segment[:, 1]))

    def test_successful_refinement_is_retained_in_connected_curve(self):
        self.payload["targets"]["refinement"] = target(.1, density=3.02, kind="refinement")
        fig = self.plot()
        self.assertEqual(len(self.segments(fig)), 1)
        self.assertTrue(np.any(np.all(self.segments(fig)[0] == [3.02, .1], axis=1)))

    def test_consistent_join_uses_new_twenty_root_once_and_no_zero_anchor(self):
        self.payload["targets"]["20.0"] = target(20., density=3.055)
        fig = self.plot()
        segments = self.segments(fig)
        self.assertEqual(len(segments), 1)
        data = segments[0]
        self.assertEqual(np.count_nonzero(data[:, 1] == 20.), 1)
        np.testing.assert_array_equal(data[data[:, 1] == 20.][0], [3.055, 20.])
        self.assertTrue(np.all(data[:, 1] >= .01))
        self.assertEqual(fig.logT_diagnostics["joins"]["50"]["status"], "joined")

    def test_inconsistent_join_has_gap_and_explicit_diagnostic(self):
        self.payload["targets"]["20.0"] = target(20., density=3.2)
        fig = self.plot()
        self.assertEqual(len(self.segments(fig)), 2)
        for segment in self.segments(fig):
            contains_cold_seam = np.any(np.all(segment == [3.2, 20.], axis=1))
            self.assertFalse(contains_cold_seam and np.max(segment[:, 1]) > 20.)
        self.assertEqual(fig.logT_diagnostics["joins"]["50"]["status"], "inconsistent")
        self.assertTrue(any("join" in text.get_text().lower() for text in fig.axes[0].texts))

    def test_unjoined_warm_run_keeps_archival_geometry_down_to_seam(self):
        for status in ("pending", "success"):
            with self.subTest(status=status):
                self.payload["targets"]["20.0"] = target(20., density=3.2, status=status)
                fig = self.plot()
                warm_run = next(segment for segment in self.segments(fig)
                                if np.max(segment[:, 1]) > 20.)
                np.testing.assert_array_equal(warm_run[0], [3.05, 20.])
                self.assertTrue(fig.logT_diagnostics["joins"]["50"]["warm_seam_is_display_interpolation"])
                roots = next(collection for collection in fig.axes[0].collections
                             if collection.get_gid() == "cold-solver-roots:50").get_offsets()
                self.assertFalse(np.any(np.all(roots == [3.05, 20.], axis=1)))

    def test_nonfinite_or_wrong_temperature_root_does_not_make_a_bridge(self):
        for bad in (dict(nB_0minus_over_n0=float("nan")), dict(T_0minus=0.)):
            with self.subTest(bad=bad):
                self.payload["targets"]["1.0"] = target(1.)
                self.payload["targets"]["1.0"]["root"].update(bad)
                for segment in self.segments(self.plot()):
                    self.assertFalse(np.min(segment[:, 1]) < 1. < np.max(segment[:, 1]))

    def test_archive_round_trip_is_portable_and_preserves_provenance(self):
        self.payload["targets"]["1.0"]["root"]["continuation_state"] = {"large": np.arange(20)}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plot.npz"
            self.module.export_plot_archive(self.warm, self.payload, path)
            with np.load(path, allow_pickle=False) as archive:
                self.assertTrue(all(archive[key].dtype != object for key in archive.files))
                metadata = json.loads(str(archive["metadata_json"]))
                self.assertIn("smoothed", metadata["warm_provenance"])
                self.assertFalse(metadata["warm_vertices_are_solver_points"])
                self.assertIn("joins", metadata)
            warm, payload = self.module.load_plot_archive(path)
            self.assertTrue(np.all(warm["curve_vertices_0"][:, 1] > 20.))
            self.assertNotIn("continuation_state", payload["targets"]["1.0"]["root"])
            self.assertEqual(len(warm["roots_0"]), 1)
            before = self.segments(self.plot())
            after = self.segments(self.module.plot_contours(warm, payload))
            self.assertEqual(len(before), len(after))
            for a, b in zip(before, after):
                np.testing.assert_array_equal(a, b)

    def test_archive_preserves_source_fingerprints_and_checkpoint_time(self):
        inputs = {"warm_archive": {"sha256": "warm-snapshot-digest"},
                  "warm_manifest": {"sha256": "manifest-digest"}}
        self.payload.update(identity={"inputs": inputs}, domain_fingerprint="domain-digest",
                            updated_unix_s=123456.75)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plot.npz"
            self.module.export_plot_archive(self.warm, self.payload, path)
            with np.load(path, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata_json"]))
                self.assertIn("source_inputs", metadata)
                self.assertEqual(metadata["source_inputs"], inputs)
                self.assertEqual(metadata["domain_fingerprint"], "domain-digest")
                self.assertEqual(metadata["checkpoint_updated_unix_s"], 123456.75)
            warm, payload = self.module.load_plot_archive(path)
            self.assertEqual(payload["identity"]["inputs"], inputs)
            self.assertEqual(payload["updated_unix_s"], 123456.75)
            fig = self.module.plot_contours(warm, payload)
            self.assertEqual(fig.logT_diagnostics["source_inputs"], inputs)
            self.assertEqual(fig.logT_diagnostics["domain_fingerprint"], "domain-digest")

    def test_failed_accuracy_check_remains_visible_in_saved_plot(self):
        self.payload['validation']={'check':dict(status='failed',relative_speed_change=.02)}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'plot.npz'
            self.module.export_plot_archive(self.warm,self.payload,path)
            warm,payload=self.module.load_plot_archive(path)
            self.assertEqual(payload['validation']['check']['status'],'failed')
            fig=self.module.plot_contours(warm,payload)
            self.assertTrue(any('Accuracy checks unresolved' in t.get_text() for t in fig.axes[0].texts))

    def test_figure_outputs_replace_completed_temporary_files_atomically(self):
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
                self.module.plot_contours(self.warm, self.payload, output_dir=tmp)
            self.assertEqual(observed, [".png", ".pdf"])
            self.assertEqual(sorted(path.name for path in Path(tmp).iterdir()),
                             ["isothermal-logT.pdf", "isothermal-logT.png"])

    def test_default_schedule_keeps_unlisted_temperatures_as_gaps(self):
        del self.payload["temperatures"]
        fig = self.plot()
        self.assertGreater(fig.logT_diagnostics["missing_base_targets"], 20)
        for segment in self.segments(fig):
            self.assertFalse(np.min(segment[:, 1]) < .5 < np.max(segment[:, 1]))

    def test_module_imports_no_solver_or_eos(self):
        source = PLOT_PATH.read_text()
        self.assertNotIn("import RMFsolver", source)
        self.assertNotIn("from RMFsolver", source)
        self.assertNotIn("solve_front_isothermal(", source)

    def test_notebook_has_independent_saved_data_section_before_gw(self):
        with (ROOT / "All_Plots_new.ipynb").open() as stream:
            notebook = json.load(stream)
        cells = notebook["cells"]
        matches = [i for i, cell in enumerate(cells)
                   if "# Numerical isothermal contour: logarithmic temperature" in "".join(cell["source"])]
        self.assertEqual(len(matches), 1, "The independent logT notebook section is missing")
        index = matches[0]
        self.assertEqual(index, 19)
        loading = "".join(cells[index + 1]["source"])
        plotting = "".join(cells[index + 2]["source"])
        self.assertIn("load_plot_archive", loading)
        self.assertIn("isothermal-logT-plot.npz", loading)
        self.assertTrue(plotting.startswith("# Tunable plotting parameters\n"))
        self.assertIn("# GW signal section", "".join(cells[index + 3]["source"]))
        self.assertNotIn("RMFsolver", loading + plotting)
        self.assertNotIn("_logT_engine", loading + plotting)
        self.assertNotIn("isothermal_contour_data", loading + plotting)


if __name__ == "__main__":
    unittest.main()
