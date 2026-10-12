"""Synthetic plotting checks only; these tests never run a numerical BVP."""
from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PLOT_PATH = ROOT / "new_paper_calculations/26-09-21/isothermal_targeted/plot_targeted_contours.py"


def fixture():
    density, temperature = np.meshgrid([2., 3., 4.], [0., 20., 40., 60.])
    angles = np.array([0., np.pi / 4., np.pi / 2.])
    domain = dict(
        domain_fingerprint="plot-fixture",
        nB_0minus_over_n0_grid=density, T_0minus_grid_MeV=temperature,
        theta_axis_rad=angles, rho_inner_boundary=np.full(3, .2),
        rho_outer_boundary=np.full(3, .95),
        ray_transform=dict(density_origin_over_n0=1., density_scale_over_n0=4.5,
                           temperature_scale_MeV=120.),
        boundary_ray_records=[dict(theta_rad=theta, delta_muB_equals_zero={"rho": .2},
                                   a_0plus_equals_one={"rho": .95}) for theta in angles],
    )
    numerical = dict(velocity_m_s_grid=(density - 2.) * 500.,
                     task_status=np.full(density.shape, "success", object))
    return domain, numerical


def point(temperature, density=3.05, level=500.):
    return dict(T_0minus=temperature, nB_0minus_over_n0=density,
                velocity_m_s=level, level=level)


def task(**changes):
    result = dict(id="test-target", level=500., T_0minus=30., status="success",
                  certified=True, left=point(15.), root=point(30.), right=point(45.), children=[])
    result.update(changes)
    return result


def payload(row=None, roots=()):
    return dict(targets={} if row is None else {row["id"]: row}, source_roots=list(roots), points={})


class TargetedPlotTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        plt.close("all")

    def setUp(self):
        self.assertTrue(PLOT_PATH.is_file(), "The targeted contour plotting module is not implemented")
        spec = importlib.util.spec_from_file_location("targeted_plot_under_test", PLOT_PATH)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.domain, self.numerical = fixture()

    def tearDown(self):
        plt.close("all")

    def plot(self, targeted=None, repair=None, **kwargs):
        return self.module.plot_targeted(self.domain, self.numerical, repair, targeted, **kwargs)

    @staticmethod
    def new_lines(ax):
        return [line for line in ax.lines if str(line.get_gid()).startswith("targeted:")]

    @staticmethod
    def marker_positions(ax):
        groups = [np.asarray(collection.get_offsets()) for collection in ax.collections
                  if str(collection.get_gid()).startswith("accepted-contour-roots:")]
        return np.concatenate(groups) if groups else np.empty((0, 2))

    def test_no_target_payload_keeps_square_style_and_baseline(self):
        fig, ax = self.plot()
        np.testing.assert_allclose(fig.get_size_inches(), [7.2, 7.2])
        self.assertEqual(ax.get_xlim(), (1., 5.5))
        self.assertEqual(ax.get_ylim(), (0., 115.))
        self.assertEqual(ax.get_box_aspect(), 1.)
        self.assertEqual(ax.xaxis.label.get_fontsize(), 18.)
        self.assertEqual(ax.yaxis.label.get_fontsize(), 18.)
        self.assertEqual(ax.title.get_fontsize(), 20.)
        self.assertEqual(len(ax.patches), 2)
        for patch in ax.patches:
            np.testing.assert_allclose(patch.get_facecolor(), to_rgba("0.72"))
        self.assertTrue(any(collection.get_gid() == "baseline-repair-contours" for collection in ax.collections))
        self.assertEqual(len(self.new_lines(ax)), 0)

    def test_only_certified_successful_leaf_tasks_add_lines(self):
        cases = [task(status=status) for status in
                 ("failed", "pending", "needs_refinement", "budget_exhausted", "missing_profile")]
        cases += [task(certified=False), task(children=["child-target"]), task(root=None), task(left=None), task(right=None)]
        for row in cases:
            with self.subTest(row=row):
                fig, ax = self.plot(payload(row))
                self.assertEqual(len(self.new_lines(ax)), 0)
                plt.close(fig)

    def test_certified_interval_connects_actual_points_despite_large_density_spacing(self):
        row = task(left=point(15., 2.8), root=point(30., 3.2), right=point(45., 3.6))
        _, ax = self.plot(payload(row))
        lines = self.new_lines(ax)
        self.assertEqual(len(lines), 1)
        np.testing.assert_allclose(lines[0].get_xydata(), [[2.8, 15.], [3.2, 30.], [3.6, 45.]])
        self.assertEqual(lines[0].get_color(), "#8c6d31")
        self.assertEqual(lines[0].get_linewidth(), 1.8)

    def test_uncertified_and_refining_roots_remain_visible_without_a_bridge(self):
        for status in ("success", "needs_refinement"):
            with self.subTest(status=status):
                fig, ax = self.plot(payload(task(status=status, certified=False)))
                self.assertEqual(len(self.new_lines(ax)), 0)
                self.assertTrue(np.any(np.all(np.isclose(self.marker_positions(ax), [3.05, 30.]), axis=1)))
                plt.close(fig)

    def test_source_and_old_successful_roots_are_colored_markers(self):
        source = point(10., 2.8, 200.)
        old = dict(point(25., 3.1), theta_rad=.4, radial_fraction=.5)
        failed = dict(point(35., 3.2), theta_rad=.5, radial_fraction=.5)
        repair = dict(domain_fingerprint="plot-fixture", controls=self.module.DEFAULT_REPAIR_CONTROLS.copy(),
                      rays={"good": dict(level=500., theta_rad=.4, roots=[old], status="success"),
                            "bad": dict(level=500., theta_rad=.5, roots=[failed], status="failed")})
        _, ax = self.plot(payload(roots=[source]), repair=repair)
        positions = self.marker_positions(ax)
        self.assertTrue(np.any(np.all(np.isclose(positions, [2.8, 10.]), axis=1)))
        self.assertTrue(np.any(np.all(np.isclose(positions, [3.1, 25.]), axis=1)))
        self.assertFalse(np.any(np.all(np.isclose(positions, [3.2, 35.]), axis=1)))
        markers = {group.get_gid(): group for group in ax.collections
                   if str(group.get_gid()).startswith("accepted-contour-roots:")}
        np.testing.assert_allclose(markers["accepted-contour-roots:200"].get_facecolors()[0],
                                   to_rgba("#ff7f0e"))

    def test_temperature_clipping_preserves_boundary_pieces_and_other_branches(self):
        original = [np.array([[3., 0.], [3., 60.]]), np.array([[4., 0.], [4., 60.]])]
        target = np.array([[3.05, 15.], [3.05, 30.], [3.05, 45.]])
        clipped = self.module._trim_covered_segments(original, [target])
        self.assertEqual(len(clipped), 3)
        np.testing.assert_allclose(clipped[0], [[3., 0.], [3., 15.]])
        np.testing.assert_allclose(clipped[1], [[3., 45.], [3., 60.]])
        np.testing.assert_allclose(clipped[2], original[1])

    def test_no_certified_coverage_does_not_change_baseline_geometry(self):
        original = [np.array([[2.7, 0.], [3., 20.], [3.2, 60.]])]
        clipped = self.module._trim_covered_segments(original, [])
        self.assertEqual(len(clipped), 1)
        np.testing.assert_array_equal(clipped[0], original[0])

    def test_disjoint_certified_intervals_do_not_bridge_the_middle_gap(self):
        original = [np.array([[3., 0.], [3., 60.]])]
        targets = [np.array([[3.05, 10.], [3.05, 20.]]), np.array([[3.05, 40.], [3.05, 50.]])]
        clipped = self.module._trim_covered_segments(original, targets)
        self.assertEqual(len(clipped), 3)
        np.testing.assert_allclose(clipped[1], [[3., 20.], [3., 40.]])

    def test_nonfinite_roots_never_make_lines_or_markers(self):
        _, ax = self.plot(payload(task(root=point(np.nan))))
        self.assertEqual(len(self.new_lines(ax)), 0)
        self.assertEqual(len(self.marker_positions(ax)), 0)

    def test_public_merged_segments_match_certified_plot_geometry(self):
        self.assertTrue(callable(getattr(self.module, "merged_targeted_segments", None)),
                        "A public geometry helper is required for the notebook")
        targeted = payload(task())
        merged = self.module.merged_targeted_segments(self.domain, self.numerical, None, targeted)
        _, ax = self.plot(targeted)
        target_line = self.new_lines(ax)[0].get_xydata()
        original = self.module.merged_contour_segments(self.domain, self.numerical, None)
        for level, before, after in zip(self.module.LEVELS, original, merged):
            expected = (self.module._trim_covered_segments(before, [target_line]) + [target_line]
                        if level == 500. else before)
            self.assertEqual(len(after), len(expected))
            for actual_line, expected_line in zip(after, expected):
                np.testing.assert_array_equal(actual_line, expected_line)

    def test_public_merged_segments_support_requested_level_subset(self):
        self.assertIn("levels", inspect.signature(self.module.merged_targeted_segments).parameters,
                      "The notebook must be able to select its displayed contour levels")
        targeted = payload(task())
        full = self.module.merged_targeted_segments(self.domain, self.numerical, None, targeted)
        levels = (200., 500., 1000.)
        subset = self.module.merged_targeted_segments(self.domain, self.numerical, None, targeted, levels=levels)
        self.assertEqual(len(subset), len(levels))
        for level, actual in zip(levels, subset):
            expected = full[self.module.LEVELS.index(level)]
            self.assertEqual(len(actual), len(expected))
            for actual_line, expected_line in zip(actual, expected):
                np.testing.assert_array_equal(actual_line, expected_line)

    def test_pending_or_unattempted_uncertified_tasks_preserve_original_geometry(self):
        self.assertTrue(callable(getattr(self.module, "merged_targeted_segments", None)),
                        "A public geometry helper is required for the notebook")
        baseline = self.module.merged_targeted_segments(self.domain, self.numerical, None)
        for row in (task(status="pending"), task(certified=False),
                    task(children=["child-target"])):
            with self.subTest(row=row):
                merged = self.module.merged_targeted_segments(self.domain, self.numerical, None, payload(row))
                for expected, actual in zip(baseline, merged):
                    self.assertEqual(len(actual), len(expected))
                    for actual_line, expected_line in zip(actual, expected):
                        np.testing.assert_array_equal(actual_line, expected_line)

    def test_unresolved_attempts_mask_inherited_crossings_without_new_edges(self):
        original = self.module.merged_contour_segments(self.domain, self.numerical, None)
        for status in ("failed", "deferred", "needs_refinement", "budget_exhausted", "missing_profile"):
            for root in (None, point(30.)):
                with self.subTest(status=status, root=root):
                    row = task(status=status, certified=False, root=root)
                    mask = np.asarray([[3.05, 15.], [3.05, 45.]] if root is None else
                                      [[3.05, 15.], [3.05, 30.], [3.05, 45.]])
                    targeted = payload(row)
                    merged = self.module.merged_targeted_segments(self.domain, self.numerical, None, targeted)
                    for level, before, after in zip(self.module.LEVELS, original, merged):
                        expected = (self.module._trim_covered_segments(before, [mask])
                                    if level == 500. else before)
                        self.assertEqual(len(after), len(expected))
                        for actual_line, expected_line in zip(after, expected):
                            np.testing.assert_array_equal(actual_line, expected_line)
                    fig, ax = self.plot(targeted)
                    self.assertEqual(len(self.new_lines(ax)), 0)
                    plt.close(fig)

    def test_refining_parent_masks_full_interval_and_certified_child_restores_only_its_piece(self):
        parent = task(status="needs_refinement", certified=False, children=["child"])
        child = task(id="child", T_0minus=22.5, left=point(15.), root=point(22.5), right=point(30.))
        targeted = payload(parent)
        targeted["targets"][child["id"]] = child
        merged = self.module.merged_targeted_segments(self.domain, self.numerical, None, targeted)
        original = self.module.merged_contour_segments(self.domain, self.numerical, None)
        index = self.module.LEVELS.index(500.)
        expected = self.module._trim_covered_segments(original[index],
                                                      [np.array([[3.05, 15.], [3.05, 30.], [3.05, 45.]])])
        expected.append(np.array([[3.05, 15.], [3.05, 22.5], [3.05, 30.]]))
        self.assertEqual(len(merged[index]), len(expected))
        for actual_line, expected_line in zip(merged[index], expected):
            np.testing.assert_array_equal(actual_line, expected_line)
        _, ax = self.plot(targeted)
        self.assertEqual([line.get_gid() for line in self.new_lines(ax)], ["targeted:child"])

    def test_public_root_groups_match_plot_markers_including_uncertified_root(self):
        self.assertTrue(callable(getattr(self.module, "accepted_root_groups", None)),
                        "A public accepted-root helper is required for the notebook")
        targeted = payload(task(certified=False), roots=[point(10., 2.8, 200.)])
        groups = self.module.accepted_root_groups(None, targeted)
        self.assertEqual(groups[200.], [(2.8, 10.)])
        self.assertEqual(groups[500.], [(3.05, 30.)])
        self.assertTrue(all(not values for values in self.module.accepted_root_groups(None).values()))
        _, ax = self.plot(targeted)
        expected = np.asarray([coordinates for values in groups.values() for coordinates in values])
        np.testing.assert_array_equal(self.marker_positions(ax), expected)

    def repair(self, entries):
        """`entries` are (theta index, radial fraction or None) on a 5-node axis."""
        axis = np.linspace(0., np.pi / 2., 5)
        domain = dict(self.domain, theta_axis_rad=axis,
                      rho_inner_boundary=np.full(axis.size, .2),
                      rho_outer_boundary=np.full(axis.size, .95))
        rays = {}
        for index, fraction in entries:
            theta = float(axis[index])
            roots = [] if fraction is None else [dict(
                theta_rad=theta, radial_fraction=fraction, velocity_m_s=500.,
                **dict(zip(("nB_0minus_over_n0", "T_0minus"),
                           self.module.ray_coordinates(domain, theta, fraction)[0])))]
            rays[f"500:{index}"] = dict(level=500., theta_rad=theta, roots=roots,
                                        status="success" if roots else "target_budget_exhausted")
        return domain, dict(rays=rays, controls={}, domain_fingerprint="plot-fixture")

    def test_a_gap_in_the_angular_schedule_is_bridged_onto_its_two_end_roots(self):
        domain, repair = self.repair([(0, .3), (1, .4), (3, .6), (4, .7)])
        bridges = self.module.targeted_bridges(domain, repair)[self.module.LEVELS.index(500.)]
        self.assertEqual([len(bridge) for bridge in bridges], [3])
        ends = [repair["rays"][key]["roots"][0] for key in ("500:1", "500:3")]
        np.testing.assert_allclose(
            bridges[0][[0, -1]],
            [[end["nB_0minus_over_n0"], end["T_0minus"]] for end in ends], atol=1e-12)

    def test_a_bridge_follows_the_domain_geometry_rather_than_the_chord(self):
        domain, repair = self.repair([(0, .1), (3, .9)])
        bridge = self.module.targeted_bridges(domain, repair)[self.module.LEVELS.index(500.)][0]
        chord = np.linspace(bridge[0], bridge[-1], len(bridge))
        self.assertGreater(np.abs(bridge - chord).max(), .1)

    def test_open_ends_and_failed_only_levels_are_never_extrapolated(self):
        domain, repair = self.repair([(1, .4), (2, .5), (3, .6)])
        self.assertEqual(self.module.targeted_bridges(domain, repair)[self.module.LEVELS.index(500.)], [])
        domain, repair = self.repair([(2, .5)])
        self.assertEqual(self.module.targeted_bridges(domain, repair)[self.module.LEVELS.index(500.)], [])

    def test_a_certified_targeted_interval_removes_the_bridge_it_covers(self):
        domain, repair = self.repair([(0, .3), (2, .5)])
        bridges = self.module.targeted_bridges(domain, repair)[self.module.LEVELS.index(500.)]
        self.assertEqual([len(bridge) for bridge in bridges], [3])
        bridge = bridges[0]
        covering = task(left=point(bridge[0][1], bridge[0][0]),
                        root=point(bridge[1][1], bridge[1][0]),
                        right=point(bridge[2][1], bridge[2][0]))
        self.assertEqual(
            self.module.targeted_bridges(domain, repair, payload(covering))[self.module.LEVELS.index(500.)], [])

    def test_bridges_are_drawn_dashed_and_are_not_part_of_the_contour_set(self):
        domain, repair = self.repair([(0, .3), (3, .6)])
        self.module.plot_targeted(domain, self.numerical, repair)
        ax = plt.gcf().axes[0]
        dashed = [line for line in ax.lines if str(line.get_gid()).startswith("interpolated-bridge:")]
        self.assertEqual(len(dashed), 1)
        self.assertNotEqual(dashed[0].get_linestyle(), "-")

    def test_a_continuous_curve_is_one_polyline_with_a_solved_node_mask(self):
        domain, repair = self.repair([(0, .3), (2, .5), (4, .7)])
        vertices, solved = self.module.continuous_level_curves(domain, repair)[
            self.module.LEVELS.index(500.)]
        self.assertEqual(len(vertices), 5)
        np.testing.assert_array_equal(solved, [True, False, True, False, True])

    def test_an_open_end_is_continued_only_when_bracketed_by_two_levels(self):
        axis = np.linspace(0., np.pi / 2., 5)
        domain = dict(self.domain, theta_axis_rad=axis,
                      rho_inner_boundary=np.full(axis.size, .2),
                      rho_outer_boundary=np.full(axis.size, .95))

        def ray(level, node, fraction):
            theta = float(axis[node])
            return dict(level=level, theta_rad=theta, status="success", roots=[dict(
                theta_rad=theta, radial_fraction=fraction, velocity_m_s=level,
                **dict(zip(("nB_0minus_over_n0", "T_0minus"),
                           self.module.ray_coordinates(domain, theta, fraction)[0])))])

        # 1000 stops at node 2; 500 below and 2000 above cover nodes 3 and 4.
        rays = [ray(500., node, .30 + .02 * node) for node in range(5)]
        rays += [ray(1000., node, .60 + .02 * node) for node in range(3)]
        rays += [ray(2000., node, .90 + .01 * node) for node in range(5)]
        repair = dict(rays={f"{r['level']:g}:{r['theta_rad']}": r for r in rays},
                      controls={}, domain_fingerprint="plot-fixture")
        curves = self.module.continuous_level_curves(domain, repair)
        _, solved = curves[self.module.LEVELS.index(1000.)]
        self.assertEqual(solved.tolist(), [True, True, True, False, False])
        # 2000 is the top level, so its own sweep is never continued past its data.
        rays = [r for r in rays if r["level"] != 2000.] + [ray(2000., node, .90 + .01 * node)
                                                           for node in range(3)]
        repair = dict(rays={f"{r['level']:g}:{r['theta_rad']}": r for r in rays},
                      controls={}, domain_fingerprint="plot-fixture")
        for level in (1000., 2000.):
            with self.subTest(level=level):
                _, solved = self.module.continuous_level_curves(domain, repair)[
                    self.module.LEVELS.index(level)]
                self.assertEqual(solved.size, 3)

    def test_resampling_keeps_every_scheduled_node_on_the_curve(self):
        domain, repair = self.repair([(0, .3), (1, .45), (2, .5), (3, .62), (4, .7)])
        index = self.module.LEVELS.index(500.)
        nodes, _ = self.module.continuous_level_curves(domain, repair)[index]
        dense, _ = self.module.continuous_level_curves(domain, repair, samples=201)[index]
        # Sampling is per node interval, so the total is near `samples`, not exact.
        self.assertGreaterEqual(len(dense), 201)
        self.assertLess(len(dense), 201 + len(nodes))
        for node in nodes:
            self.assertLess(np.abs(dense - node).sum(axis=1).min(), 1e-9)
        # Every interval gets the same number of samples despite the geometric axis.
        counts = [int(np.sum((dense[:, 1] >= first[1]) & (dense[:, 1] <= second[1])))
                  for first, second in zip(nodes[:-1], nodes[1:])]
        self.assertEqual(len(set(counts)), 1)

    def test_extra_checkpoints_contribute_roots_without_touching_the_repair_payload(self):
        domain, repair = self.repair([(0, .3), (4, .7)])
        index = self.module.LEVELS.index(500.)
        _, solved = self.module.continuous_level_curves(domain, repair)[index]
        self.assertEqual(solved.sum(), 2)
        before = {key: dict(row) for key, row in repair["rays"].items()}
        _, extra = self.repair([(2, .5)])
        _, solved = self.module.continuous_level_curves(domain, repair, extra=(extra,))[index]
        self.assertEqual(solved.sum(), 3)
        self.assertEqual({key: dict(row) for key, row in repair["rays"].items()}, before)

    def test_the_cold_anchor_starts_every_curve_at_three_n0_and_zero_temperature(self):
        domain, repair = self.repair([(2, .5), (4, .7)])
        index = self.module.LEVELS.index(500.)
        plain, _ = self.module.continuous_level_curves(domain, repair)[index]
        self.assertGreater(plain[0][1], 0.)
        anchored, solved = self.module.continuous_level_curves(
            domain, repair, cold_anchor=True)[index]
        # The fixture's inner boundary, i.e. its own analogue of (3 n0, 0).
        corner = self.module.ray_coordinates(domain, [0.], [0.])[0]
        np.testing.assert_allclose(anchored[0], corner, atol=1e-12)
        self.assertEqual(anchored[0][1], 0.)
        # The anchor is not a solved root, and it adds no intermediate node of
        # its own: node 3 here is the pre-existing interior fill between 2 and 4.
        self.assertEqual(solved.tolist(), [False, True, False, True])
        np.testing.assert_allclose(anchored[1:], plain, atol=1e-12)

    def test_the_cold_anchor_is_skipped_when_a_level_already_reaches_zero_angle(self):
        domain, repair = self.repair([(0, .3), (2, .5)])
        index = self.module.LEVELS.index(500.)
        plain, _ = self.module.continuous_level_curves(domain, repair)[index]
        anchored, solved = self.module.continuous_level_curves(
            domain, repair, cold_anchor=True)[index]
        np.testing.assert_array_equal(anchored, plain)
        self.assertEqual(solved.tolist(), [True, False, True])

    def test_targeted_plot_saves_png_and_pdf_without_temporary_leftovers(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "plots"
            self.plot(payload(task()), output_dir=output)
            self.assertEqual({path.name for path in output.iterdir()},
                             {"isothermal-targeted-contours.png", "isothermal-targeted-contours.pdf"})
            for extension in ("png", "pdf"):
                self.assertGreater((output / f"isothermal-targeted-contours.{extension}").stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
