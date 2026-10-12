"""Fast orchestration/continuation tests; no production BVP scan is run here."""
import copy
import inspect
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "new_paper_calculations/26-09-16/isothermal_numeric_repair"
sys.path.insert(0, str(WORKFLOW))
import _repair_common as common
from _repair_search import Repair, brackets, connected_segments, current_bracket, point_id, within_spacing
from RMFsolver import phase_velocity as pv


def fixture(speed=None, theta_count=3):
    speed = speed or (lambda theta, r: 1000. * r / (1. - r))
    theta = np.array([0., .5, np.pi / 2.]) if theta_count == 3 else np.linspace(0., np.pi / 2., theta_count)
    fractions = np.array([.02, .2, .5, .99])
    rho = .4 + fractions[None, :] * .5 + np.zeros((theta.size, 1))
    x = 1. + 4.5 * rho * np.cos(theta[:, None])
    temperature = np.maximum(.01, 120. * rho * np.sin(theta[:, None]))
    velocity = np.array([[speed(t, r) for r in fractions] for t in theta])
    currents = velocity / common.C * x * 1.e6
    domain = dict(theta_axis_rad=theta, radial_fraction_axis=fractions, domain_fingerprint="fixture",
                  nB_0minus_grid=x * 1.e6, nB_0minus_over_n0_grid=x, T_0minus_grid_MeV=temperature,
                  rho_inner_boundary=np.full(theta.size, .4), rho_outer_boundary=np.full(theta.size, .9),
                  ray_transform=dict(density_origin_over_n0=1., density_scale_over_n0=4.5,
                                     temperature_scale_MeV=120., temperature_min_MeV=.01),
                  B_one_forth_MeV=189.1565957288247, ms_MeV=0., NM_type="PNM", upB=5000, xi=-.5)
    records = np.empty((theta.size, 4), object)
    for ij in np.ndindex(records.shape):
        records[ij] = dict(success=True, jB=float(currents[ij]), velocity_m_s=float(velocity[ij]), immutable="original")
    numerical = dict(task_status=np.full((theta.size, 4), "success", object), records=records,
                     velocity_m_s_grid=velocity)
    analytic = dict(jB_grid=currents.copy(), velocity_m_s_grid=velocity.copy())
    return [domain, analytic, numerical]


def fake_trial(**kwargs):
    point = kwargs["point"]
    r = point["radial_fraction"]
    velocity = 1000. * r / (1. - r)
    jB = velocity / common.C * point["nB_0minus"]
    return dict(success=True, status="success", velocity_m_s=velocity, jB=jB,
                continuation_state=dict(s=np.array([0., .5, .999]), nK=np.array([1.e6, 4.e5, 1.e2]),
                                        jK=np.array([jB, .4*jB, .0001*jB]), nK_inf=0., jK_inf=0., jB=jB))


def sleeping_child(connection, kwargs):
    time.sleep(5.)


class SolverContinuationTests(unittest.TestCase):
    def seed(self):
        return dict(s=np.array([0., .01, .1, .999]), nK=np.array([10., 9., 5., .01]),
                    jK=np.array([4., 3., 2., .004]), nK_inf=0., jK_inf=0., jB=4.)

    def state(self):
        return dict(nK_inf=1., nK_0plus=21., jK_inf=.5, jK_0plus=8.5, tail_coefficient=.4)

    def test_transfer_preserves_adapted_mesh_and_endpoint_scaling(self):
        seed = self.seed()
        before = copy.deepcopy(seed)
        mesh, nK, jK = pv._isothermal_continuation_initial_guess(seed, self.state(), .9999, 8, 100)
        self.assertTrue(np.all(np.diff(mesh) > 0))
        for node in seed["s"] * (.9999 / .999):
            self.assertIn(node, mesh)
        self.assertEqual(nK[0], 21.)
        self.assertEqual(jK[0], 8.5)
        self.assertAlmostEqual(jK[-1], .5 + .4 * (nK[-1] - 1.))
        np.testing.assert_array_equal(seed["nK"], before["nK"])

    def test_invalid_profiles_rejected(self):
        for field in ("s", "nK", "jK"):
            seed = self.seed(); seed[field][1] = np.nan
            with self.assertRaises(ValueError):
                pv._isothermal_continuation_initial_guess(seed, self.state(), .999, 8, 100)
        with self.assertRaises(ValueError):
            pv._isothermal_continuation_initial_guess(self.seed(), self.state(), .999, 8, 4)

    def test_optional_api_preserves_old_defaults(self):
        signature = inspect.signature(pv.solve_front_isothermal)
        self.assertIsNone(signature.parameters["continuation_guess"].default)
        self.assertFalse(signature.parameters["return_continuation"].default)
        self.assertEqual(signature.parameters["tail_eps"].default, .001)
        self.assertEqual(signature.parameters["NM_type"].default, "PNM")

    @unittest.skipUnless(os.environ.get("NSM_ISOTHERMAL_LIVE_TEST") == "1", "Opt-in bounded single-core live check")
    def test_live_lightweight_continuation_matches_original_warm_speed(self):
        import signal
        from RMFsolver import constants as const
        def timeout(signum, frame):
            raise TimeoutError("Bounded warm continuation check exceeded 45 seconds")
        handler = signal.signal(signal.SIGALRM, timeout)
        signal.alarm(45)
        try:
            args = dict(T=10., nB_0minus=2. * .16 * const.MeV_fm**3, B_one_forth=180., a_0plus=.6,
                        n_mesh=80, tol_bvp=1.e-3, max_nodes=3000, jB_guess=.5,
                        jB_bounds=(1.e-4, 5.), return_continuation=True)
            first = pv.solve_front_isothermal(**args)
            self.assertTrue(first["success"], first.get("message"))
            self.assertEqual(first["profile_state_calls"], 0)
            second = pv.solve_front_isothermal(**dict(args, jB_guess=first["jB"]),
                                               continuation_guess=first["continuation_state"])
            self.assertTrue(second["success"], second.get("message"))
            self.assertTrue(second["continuation_used"])
            self.assertEqual(second["profile_state_calls"], 0)
            self.assertLess(abs(second["jB"] / first["jB"] - 1.), 1.e-3)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, handler)


class RepairTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.baseline = fixture()
        self.repair = Repair(self.directory, self.baseline, {"baseline": "unchanged"},
                             trial=fake_trial, solver_identity={"test": True})

    def tearDown(self):
        self.temp.cleanup()

    def test_local_current_bracket_never_clips_analytic_seed(self):
        lo, hi = current_bracket(5., 6.38e4, np.nan)
        self.assertEqual((lo, hi), (1.25, 4. * 6.38e4))
        self.assertGreater(hi, 6.38e4)
        with self.assertRaises(ValueError):
            current_bracket(None, np.nan, -1.)

    def test_accepted_baseline_never_changed(self):
        before = copy.deepcopy(self.baseline[2])
        point = self.repair.evaluate(.5, .2, time.monotonic() + 10., anchor=True)
        self.assertIsNotNone(point)
        np.testing.assert_array_equal(self.baseline[2]["velocity_m_s_grid"], before["velocity_m_s_grid"])
        self.assertEqual(self.baseline[2]["records"].tolist(), before["records"].tolist())
        self.assertEqual(self.repair.payload["baseline_success_count"], 12)

    def test_save_happens_before_progress_for_every_trial(self):
        observed = []
        path = self.repair.output
        class Progress:
            def update(self, count):
                payload = common.load(path)
                observed.append(len(payload["evaluations"]))
        self.repair.progress = Progress()
        self.repair.evaluate(.5, .2, time.monotonic() + 10., anchor=True)
        self.assertEqual(observed, [1])
        point = next(iter(common.load(path)["points"].values()))
        self.assertTrue((self.directory / point["profile_file"]).exists())

    def test_resume_reuses_success_without_another_trial(self):
        self.repair.evaluate(.5, .2, time.monotonic() + 10., anchor=True)
        def forbidden(**kwargs):
            raise AssertionError("Successful point must be cached")
        resumed = Repair(self.directory, self.baseline, {"baseline": "unchanged"},
                         trial=forbidden, solver_identity={"test": True})
        self.assertIsNotNone(resumed.evaluate(.5, .2, time.monotonic() + 10.))
        self.assertEqual(len(resumed.payload["evaluations"]), 1)

    def test_resume_rejects_physics_controls_source_or_fingerprint_changes(self):
        for kwargs in (dict(controls={"tail_eps": .01}), dict(solver_identity={"test": False})):
            with self.assertRaises(ValueError):
                Repair(self.directory, self.baseline, {"baseline": "unchanged"}, **kwargs)
        with self.assertRaises(ValueError):
            Repair(self.directory, self.baseline, {"baseline": "other"}, solver_identity={"test": True})
        bad = copy.deepcopy(self.baseline); bad[0]["domain_fingerprint"] = "changed"
        with self.assertRaises(ValueError):
            Repair(self.directory, bad, {"baseline": "unchanged"}, solver_identity={"test": True})

    def test_anchor_discrepancy_cannot_replace_or_seed_success(self):
        def different(**kwargs):
            out = fake_trial(**kwargs); out["velocity_m_s"] *= 1.1
            return out
        self.repair.trial = different
        self.assertIsNone(self.repair.evaluate(.5, .2, time.monotonic() + 10., anchor=True))
        self.assertEqual(self.repair.payload["evaluations"][-1]["result"]["status"], "anchor_discrepancy")
        self.assertFalse(self.repair.payload["points"])

    def test_numerical_contour_root_satisfies_actual_speed(self):
        row = self.repair.solve_ray(500., .5)
        self.assertEqual(row["status"], "success")
        self.assertEqual(len(row["roots"]), 1)
        self.assertLess(abs(row["roots"][0]["velocity_m_s"] / 500. - 1.), .005)
        self.assertGreater(len(self.repair.payload["evaluations"]), 0)

    def test_inward_and_outward_refinement_exceeds_original_sampling(self):
        for level, expected in ((5., "in"), (200000., "out")):
            row = self.repair.solve_ray(level, .5)
            self.assertEqual(row["status"], "success", row)
            r = row["roots"][0]["radial_fraction"]
            self.assertTrue(r < .02 if expected == "in" else r > .99)

    def test_two_observed_branches_are_preserved(self):
        speed = lambda theta, r: 200. + 1200. * np.sin(np.pi * r)**2
        baseline = fixture(speed)
        with tempfile.TemporaryDirectory() as path:
            def trial(**kwargs):
                out = fake_trial(**kwargs)
                point = kwargs["point"]
                v = speed(point["theta_rad"], point["radial_fraction"])
                out.update(velocity_m_s=v, jB=v * point["nB_0minus"] / common.C)
                out["continuation_state"]["jB"] = out["jB"]
                return out
            repair = Repair(path, baseline, {"baseline": "branches"}, trial=trial, solver_identity={"test": True})
            row = repair.solve_ray(500., .5)
            self.assertEqual(row["status"], "success")
            self.assertEqual(len(row["roots"]), 2)

    def test_failed_solver_never_becomes_an_interpolated_contour_point(self):
        self.repair.trial = lambda **kwargs: dict(success=False, status="trial_timeout")
        row = self.repair.solve_ray(500., .5)
        self.assertNotEqual(row["status"], "success")
        self.assertFalse(row["roots"])
        self.assertFalse(self.repair.payload["points"])

    def test_failed_ray_and_large_spacing_split_lines(self):
        def point(theta):
            return dict(theta_rad=theta, radial_fraction=.3, velocity_m_s=500.,
                        nB_0minus_over_n0=2. + theta / 100., T_0minus=50. + theta / 100.)
        rows = [dict(level=500., theta_rad=t, status="success", roots=[point(t)]) for t in range(5)]
        rows[2].update(status="trial_timeout", roots=[])
        segments = connected_segments(rows, 500., common.DEFAULT_CONTROLS)
        self.assertEqual([[p["theta_rad"] for p in line] for line in segments], [[0, 1], [3, 4]])
        rows[2].update(status="success", roots=[point(2)])
        rows[2]["roots"][0]["T_0minus"] = 100.
        self.assertEqual(len(connected_segments(rows, 500., common.DEFAULT_CONTROLS)), 2)

    def test_coarse_angular_sweep_still_connects_and_a_skipped_node_does_not(self):
        """Ray spacing wider than the continuation step is a line, not a gap."""
        axis = np.linspace(0., np.pi / 2., 20)
        def point(theta, fraction):
            return dict(theta_rad=theta, radial_fraction=fraction, velocity_m_s=500.,
                        nB_0minus_over_n0=1. + 4. * np.cos(theta), T_0minus=100. * np.sin(theta))
        fractions = np.linspace(.1, .9, axis.size)
        rows = [dict(level=500., theta_rad=float(t), status="success", roots=[point(t, float(f))])
                for t, f in zip(axis, fractions)]
        # Neighbouring roots here are far apart in (density, temperature) terms.
        self.assertFalse(within_spacing(rows[0]["roots"][0], rows[1]["roots"][0], common.DEFAULT_CONTROLS))
        segments = connected_segments(rows, 500., common.DEFAULT_CONTROLS, theta_axis=axis)
        self.assertEqual([len(line) for line in segments], [axis.size])
        # Dropping one scheduled node splits the sweep there and nowhere else.
        without = rows[:7] + rows[8:]
        segments = connected_segments(without, 500., common.DEFAULT_CONTROLS, theta_axis=axis)
        self.assertEqual([len(line) for line in segments], [7, axis.size - 8])
        # The axis is recovered from the full ray list when it is not supplied.
        self.assertEqual([len(line) for line in connected_segments(rows, 500., common.DEFAULT_CONTROLS)],
                         [axis.size])

    def test_cold_spacing_uses_temperature_ratio(self):
        first = dict(nB_0minus_over_n0=3., T_0minus=.01)
        second = dict(nB_0minus_over_n0=3., T_0minus=.02)
        self.assertFalse(within_spacing(first, second, common.DEFAULT_CONTROLS))

    def test_concurrent_requests_share_one_anchor(self):
        from concurrent.futures import ThreadPoolExecutor
        def slow(**kwargs):
            time.sleep(.01)
            return fake_trial(**kwargs)
        self.repair.trial = slow
        with ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(self.repair.evaluate, .5, .2, time.monotonic() + 5., anchor=True) for _ in range(2)]
            self.assertTrue(all(f.result() is not None for f in futures))
        self.assertEqual(len(self.repair.payload["evaluations"]), 1)

    def test_bound_expansion_is_limited_and_provisional_point_is_not_published(self):
        anchor = self.repair.evaluate(.5, .2, time.monotonic() + 5., anchor=True)
        calls = []
        def at_bound(**kwargs):
            calls.append(kwargs["bounds"])
            out = fake_trial(**kwargs)
            if len(calls) == 1:
                out["jB"] = kwargs["bounds"][0] * 1.000001
            else:
                self.assertNotIn(point_id(.5, .3), self.repair.payload["points"])
            return out
        self.repair.trial = at_bound
        self.assertIsNotNone(self.repair.evaluate(.5, .3, time.monotonic() + 5., seed=anchor))
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1], (calls[0][0] / 4., calls[0][1] * 4.))

    def test_stage_order_and_completion_do_not_claim_complete_contours(self):
        batches = []
        with patch.object(self.repair, "_batch", side_effect=lambda tasks, workers: batches.append(tasks)):
            result = self.repair.run(2)
        self.assertEqual({level for level, angle in batches[0]}, {500., 1000.})
        self.assertTrue(result["run_complete"])
        self.assertFalse(result["contours_complete"])

    def test_analytical_predictor_is_never_used_as_numerical_output(self):
        self.repair.analytic["velocity_m_s_grid"] *= 3.
        row = self.repair.solve_ray(500., .5)
        self.assertEqual(row["status"], "success")
        self.assertLess(abs(row["roots"][0]["radial_fraction"] - 1./3.), .003)

    def test_hash_validation_detects_coordinate_tampering(self):
        directory = ROOT / "new_paper_calculations/26-08-27"
        domain = common.load(directory / "isothermal-domain.npy")
        before = common.domain_fingerprint(domain)
        domain["T_0minus_grid_MeV"][0, 0] += 1.
        self.assertNotEqual(before, common.domain_fingerprint(domain))

    def test_smoke_orchestration_and_tighter_validation_resume(self):
        with tempfile.TemporaryDirectory() as path:
            baseline = fixture(theta_count=30)
            repair = Repair(path, baseline, {"fixture": "smoke"}, trial=fake_trial, solver_identity={"test": True})
            payload = repair.run(2, smoke=True)
            self.assertEqual(len(payload["rays"]), 4)
            self.assertTrue(all(ray["status"] == "success" for ray in payload["rays"].values()))
            self.assertTrue(payload["run_complete"])
            self.assertFalse(payload["contours_complete"])
            repair.validate_representatives()
            self.assertTrue(payload["validation_complete"])
            self.assertEqual(len(payload["validation"]), 4)
            count = len(payload["evaluations"])
            def forbidden(**kwargs):
                raise AssertionError("A finished smoke must resume without further numerical solves")
            resumed = Repair(path, baseline, {"fixture": "smoke"}, trial=forbidden, solver_identity={"test": True})
            resumed.run(2, smoke=True)
            resumed.validate_representatives()
            self.assertEqual(len(resumed.payload["evaluations"]), count)

    def test_hard_timeout_terminates_worker_and_returns_structured_failure(self):
        import multiprocessing as mp
        before = {p.pid for p in mp.active_children()}
        with patch.object(common, "_child", sleeping_child):
            result = common.hard_trial(timeout=.01)
        self.assertEqual(result["status"], "trial_timeout")
        self.assertFalse(result["success"])
        self.assertEqual({p.pid for p in mp.active_children()}, before)


class RealPayloadReadOnlyTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("NSM_ISOTHERMAL_LIVE_TEST") == "1", "Opt-in bounded single-core worker check")
    def test_live_spawned_worker_on_one_saved_warm_anchor(self):
        baseline, _ = common.load_baseline(ROOT / "new_paper_calculations/26-08-27")
        domain, _, numerical = baseline
        i, j = 20, 5
        old = numerical["records"][i, j]
        point = dict(T_0minus=float(domain["T_0minus_grid_MeV"][i, j]), nB_0minus=float(domain["nB_0minus_grid"][i, j]))
        physics = {key: domain[key] for key in ("B_one_forth_MeV", "ms_MeV", "NM_type", "upB")}
        result = common.hard_trial(timeout=30., point=point, physics=physics, controls=common.DEFAULT_CONTROLS,
                                   jB_guess=old["jB"], bounds=common.current_bracket(old["jB"]), continuation=None)
        self.assertTrue(result.get("success"), result.get("message"))
        self.assertLess(abs(result["velocity_m_s"] / old["velocity_m_s"] - 1.), .01)
        self.assertLess(result["endpoint_flux_residual_max"], 1.e-7)
        self.assertEqual(result["profile_state_calls"], 0)
        self.assertIn("continuation_state", result)

    def test_baseline_api_and_hashes_and_plot_shading(self):
        directory = ROOT / "new_paper_calculations/26-08-27"
        baseline, hashes = common.load_baseline(directory)
        self.assertEqual(int(np.sum(baseline[2]["task_status"] == "success")), 362)
        self.assertEqual(hashes, {name: common.sha256(directory / name) for name in common.INPUT_NAMES})
        from plot_isothermal_contours import plot_contours
        import matplotlib.pyplot as plt
        fig, ax = plot_contours(baseline[0], baseline[2])
        self.assertTrue(ax.patches[1].get_path().contains_point((1.1, 114.)))
        self.assertTrue(ax.patches[1].get_path().contains_point((5.4, 114.)))
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
