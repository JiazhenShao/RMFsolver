"""Isothermal contour workflow of 2026-10-27 (high-T polar mesh, shared engine)."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "new_paper_calculations/26-10-27/isothermal"
sys.path[:0] = [str(WORKFLOW), str(WORKFLOW / "highT")]

from RMFsolver import phase_velocity as pv  # noqa: E402

SAVED_DOMAIN = ROOT / "new_paper_calculations/26-09-16/isothermal_numeric_repair/isothermal-domain.npy"


class BoundaryTests(unittest.TestCase):
    def test_polar_round_trip(self):
        import _iso_common as common
        rho = np.array([0.1, 0.5, 0.93])
        theta = np.array([0.05, 0.8, np.pi / 2])
        x, T = common.to_physical(rho, theta)
        back_rho, back_theta = common.to_polar(x, T)
        np.testing.assert_allclose(back_rho, rho, rtol=1e-12)
        np.testing.assert_allclose(back_theta, theta, rtol=1e-12)
        self.assertEqual(float(common.to_physical(0.7, np.pi / 2)[0]), 1.0)

    def test_boundary_status_on_each_side(self):
        import _iso_common as common
        import _iso_boundaries as boundaries
        expected = (("stable_neutron_matter", "moving_front"),
                    ("moving_front", "composition_ceiling_saturated"))
        for degrees in (10.0, 45.0, 80.0):
            ray = boundaries.ray_boundaries(np.radians(degrees))
            for rho, (inside, outside) in zip((ray["rho_inner"], ray["rho_outer"]), expected):
                for factor, status in ((1 - 1e-4, inside), (1 + 1e-4, outside)):
                    x, T = common.to_physical(rho * factor, ray["theta"])
                    result = pv.analytic_velocity_isothermal_slow(
                        float(T), float(x) * common.N0_MEV3, common.PHYSICS["B_one_forth"], xi=common.XI)
                    with self.subTest(degrees=degrees, factor=factor):
                        self.assertEqual(result["status"], status)

    def test_boundaries_match_probe(self):
        import _iso_boundaries as boundaries
        window = boundaries.fixed_T_window(10.0)
        self.assertEqual(window["status"], "open")
        self.assertTrue(2.75 < window["x_inner"] < 3.0, window)
        self.assertTrue(5.25 < window["x_outer"] < 5.5, window)
        self.assertEqual(boundaries.fixed_T_window(97.0)["status"], "closed")

    def test_boundaries_cross_check_saved_domain(self):
        import _iso_common as common
        import _iso_boundaries as boundaries
        saved = np.load(SAVED_DOMAIN, allow_pickle=True).item()
        for kind, prefix in (("rho_inner", "lower_phase"), ("rho_outer", "upper_phase")):
            x = np.asarray(saved[prefix + "_nB_0minus"], dtype=float)
            x = x / common.N0_MEV3 if np.max(x) > 100 else x
            saved_rho, saved_theta = common.to_polar(x, np.asarray(saved[prefix + "_T_0minus_MeV"], dtype=float))
            for index in (3, 8, 12, 16, 20, 23):
                ray = boundaries.ray_boundaries(float(saved_theta[index]))
                with self.subTest(kind=kind, index=index):
                    self.assertLessEqual(abs(ray[kind] - saved_rho[index]), 1e-3)

    def test_boundary_curves_cover_both_figures(self):
        import _iso_boundaries as boundaries
        curves = boundaries.boundary_curves(n_rows=6, n_rays=8, workers=4)
        for name, top in (("inner", 60.0), ("outer", 94.0)):
            x, T = curves[name]
            self.assertLessEqual(T.min(), 0.0100001)
            self.assertGreaterEqual(T.max(), top)
            self.assertTrue(np.all(np.diff(T) > 0))
            self.assertTrue(np.all(np.diff(x) < 0), name)


class PolarMeshTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import polar_mesh
        cls.mesh = polar_mesh.build_mesh(workers=4)

    def test_mesh_is_strictly_inside_ring(self):
        mesh = self.mesh
        inner = np.array([b["rho_inner"] for b in mesh["boundaries"]])[:, None]
        outer = np.array([b["rho_outer"] for b in mesh["boundaries"]])[:, None]
        pending = mesh["point_status"] == "pending"
        self.assertTrue(np.all((mesh["rho"] > inner)[pending]))
        self.assertTrue(np.all((mesh["rho"] < outer)[pending]))

    def test_mesh_panel_mask(self):
        mesh = self.mesh
        pending = mesh["point_status"] == "pending"
        self.assertTrue(np.all((mesh["T"][pending] >= 8.0) & (mesh["T"][pending] <= 100.0)))
        outside = mesh["point_status"] == "outside_panel"
        self.assertTrue(np.all((mesh["T"][outside] < 8.0) | (mesh["T"][outside] > 100.0)))

    def test_mesh_counts(self):
        self.assertEqual(self.mesh["rho"].shape, (81, 29))
        self.assertGreaterEqual(np.sum(self.mesh["point_status"] == "pending"), 2000)

    def test_ninety_degree_ray_is_n0_edge(self):
        self.assertEqual(self.mesh["theta"][-1], np.pi / 2)
        np.testing.assert_array_equal(self.mesh["x"][-1], 1.0)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        import _iso_evaluate as evaluation
        self.evaluation = evaluation

    def numerical_once(self):
        if not hasattr(EvaluationTests, "_numerical"):
            import _iso_common as common
            EvaluationTests._numerical = pv.solve_front_isothermal_slow(
                10.0, 3.5 * common.N0_MEV3, common.PHYSICS["B_one_forth"], return_profile=True,
                **common.NUMERICAL)
        return dict(EvaluationTests._numerical)

    def test_reference_points(self):
        for method, speed in (("analytic", 221.7), ("numerical", 154.5)):
            record, profile = self.evaluation.evaluate(method, 3.5, 10.0)
            with self.subTest(method=method):
                self.assertEqual(record["status"], "accepted", record["message"])
                self.assertAlmostEqual(record["speed_m_s"], speed, delta=0.1)
                self.assertEqual(profile is None, method == "analytic")

    def test_analytic_intermediates_reproduce_speed(self):
        import _iso_common as common
        record, _ = self.evaluation.evaluate("analytic", 4.0, 30.0)
        a = record["a_0plus"]
        speed = common.C_M_S * np.sqrt(2 * record["I2"] / (record["lambda_n"]**2 * (1 - a)
                                                           * (1 + record["xi_used"] * a)))
        self.assertLess(abs(speed / record["speed_m_s"] - 1), 1e-12)

    def test_numerical_profile_integrals(self):
        numerical, profile = self.evaluation.evaluate("numerical", 3.5, 10.0)
        analytic, _ = self.evaluation.evaluate("analytic", 3.5, 10.0)
        self.assertLess(abs(numerical["I2_num"] / analytic["I2"] - 1), 0.3)
        self.assertTrue(-1 < numerical["xi_profile"] < 1, numerical["xi_profile"])
        self.assertEqual(len(profile["x"]), 256)

    def test_roundoff_inside_mesh_is_masked_not_failed(self):
        with patch.object(pv, "analytic_velocity_isothermal_slow",
                          return_value=dict(success=True, front_exists=False, message="",
                                            status="composition_ceiling_saturated")):
            record, _ = self.evaluation.evaluate("analytic", 5.2, 10.0)
        self.assertEqual(record["status"], "boundary_roundoff")
        with patch.object(pv, "solve_front_isothermal_slow",
                          return_value=dict(success=False, status="thermodynamic_path_failed",
                                            message="Automatic interface ceiling is not interior")):
            record, _ = self.evaluation.evaluate("numerical", 5.2, 10.0)
        self.assertEqual(record["status"], "boundary_roundoff")
        self.assertEqual(record["attempts"], 1)

    def test_nan_rms_collocation_retries_then_accepts(self):
        good = self.numerical_once()
        calls = []
        def fake(*args, **kwargs):
            calls.append(kwargs)
            return dict(good, bvp_max_rms_residual=np.nan) if len(calls) == 1 else good
        with patch.object(pv, "solve_front_isothermal_slow", side_effect=fake):
            record, _ = self.evaluation.evaluate("numerical", 3.5, 10.0)
        self.assertEqual(record["status"], "accepted")
        self.assertEqual(record["attempts"], 2)
        self.assertEqual(calls[1]["n_mesh"], 600)

    def test_shooting_branch_requires_flux_change_and_mismatch(self):
        good = self.numerical_once()
        for change, expected in ((2e-6, "unresolved"), (1e-9, "accepted")):
            shot = dict(good, solution_method="shooting", bvp_max_rms_residual=np.nan,
                        shooting_flux_change=change, shooting_mismatch=1e-10)
            with patch.object(pv, "solve_front_isothermal_slow", return_value=shot):
                record, _ = self.evaluation.evaluate("numerical", 3.5, 10.0)
            with self.subTest(change=change):
                self.assertEqual(record["status"], expected)

    def test_table_failure_uses_tolerance_ladder(self):
        good = self.numerical_once()
        calls = []
        def table(*args, **kwargs):
            calls.append(kwargs["path_rtol"])
            if kwargs["path_rtol"] > 1.5e-7:
                return dict(success=False, status="thermodynamic_path_failed",
                            message="Slow-front EOS/diffusion table did not converge")
            return good
        with patch.object(pv, "solve_front_isothermal_slow", side_effect=table):
            record, _ = self.evaluation.evaluate("numerical", 3.5, 10.0)
        self.assertEqual(record["status"], "accepted")
        np.testing.assert_allclose(calls, [1e-6, 5e-7, 2e-7, 1e-7])
        self.assertEqual(record["path_rtol_used"], 1e-7)

    def test_repair_variants_follow_the_standard_attempts(self):
        good = self.numerical_once()
        calls = []
        def picky(*args, **kwargs):
            calls.append(kwargs)
            if kwargs.get("initial_guess_mode") == "shooting" and kwargs["path_rtol"] == 2e-7:
                return good
            return dict(success=False, status="profile_rejected", message="BVP left the physical monotone path")
        with patch.object(pv, "solve_front_isothermal_slow", side_effect=picky):
            record, _ = self.evaluation.evaluate_repair("numerical", 3.5, 10.0)
        self.assertEqual(record["status"], "accepted")
        self.assertEqual(record["repair_variant"], dict(initial_guess_mode="shooting", path_rtol=2e-7))
        self.assertEqual(len(calls), 3)   # standard attempt, 600-node retry, first variant

    def test_repair_tries_tighter_tables_last(self):
        good = self.numerical_once()
        calls = []
        def tables(*args, **kwargs):
            calls.append(kwargs)
            if kwargs["path_rtol"] == 1e-7 and kwargs["initial_guess_mode"] == "auto":
                return good
            return dict(success=False, status="profile_rejected", message="Backward-shooting flux is not converged in its tolerances")
        with patch.object(pv, "solve_front_isothermal_slow", side_effect=tables):
            record, _ = self.evaluation.evaluate_repair("numerical", 3.5, 10.0)
        self.assertEqual(record["status"], "accepted")
        self.assertEqual(record["repair_variant"], dict(path_rtol=1e-7))
        self.assertEqual(calls[-1]["path_rtol"], 1e-7)

    def test_two_attempt_cap(self):
        calls = []
        def failing(*args, **kwargs):
            calls.append(kwargs)
            return dict(success=False, status="bvp_failed", message="singular Jacobian")
        with patch.object(pv, "solve_front_isothermal_slow", side_effect=failing):
            record, _ = self.evaluation.evaluate("numerical", 3.5, 10.0)
        self.assertEqual(record["status"], "unresolved")
        self.assertEqual(len(calls), 2)
        self.assertEqual(record["message"], "singular Jacobian")

    def test_acceptance_rejects_nonfinite_metrics(self):
        good = self.numerical_once()
        self.assertEqual(self.evaluation.numerical_acceptance(good, 1e-4), "")
        conservation = dict(good["conservation_errors"])
        conservation[sorted(conservation)[-1]] = np.nan
        broken = (dict(closure_error_max=np.nan), dict(conservation_errors=conservation),
                  dict(boundary_residuals=[0.0, np.nan]), dict(reaction_residual_norm=np.nan),
                  dict(diffusion_integral_residual_norm=np.nan))
        for change in broken:
            with self.subTest(change=sorted(change)):
                self.assertNotEqual(self.evaluation.numerical_acceptance(dict(good, **change), 1e-4), "")


def synthetic_mesh():
    x = np.array([[1.5, 2.0, 2.5, 3.0], [1.2, 1.4, 1.6, 1.8], [1.0, 1.0, 1.0, 1.0]])
    T = np.array([[5.0, 9.0, 12.0, 15.0], [30.0, 35.0, 40.0, 45.0], [70.0, 75.0, 80.0, 85.0]])
    status = np.where(T >= 8.0, "pending", "outside_panel").astype(object)
    return dict(theta=np.array([0.1, 0.8, np.pi / 2]), fractions=np.array([.1, .4, .7, .9]),
                x=x, T=T, rho=np.zeros_like(x), point_status=status, boundaries=[],
                fingerprint="synthetic-test-mesh")


class EngineTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        import _iso_engine as engine
        self.engine = engine
        self.directory = Path(tempfile.mkdtemp())
        self.path = self.directory / "payload.npy"

    def run_engine(self, **kwargs):
        options = dict(mesh=synthetic_mesh(), out_path=self.path, method="numerical", workers=2,
                       evaluator="tests._iso_fake_evaluator:evaluate", timeout_s=10.0)
        options.update(kwargs)
        return self.engine.run(**options)

    def test_run_fills_only_pending_points(self):
        payload = self.run_engine()
        pending = synthetic_mesh()["point_status"] == "pending"
        self.assertTrue(np.all(np.isfinite(payload["speed_m_s"][pending])))
        self.assertTrue(np.all(np.isnan(payload["speed_m_s"][~pending])))
        self.assertEqual(payload["new_evaluations"], int(pending.sum()))

    def test_resume_skips_completed_points(self):
        self.run_engine()
        self.assertEqual(self.run_engine()["new_evaluations"], 0)

    def test_resume_rejects_changed_identity(self):
        self.run_engine()
        with self.assertRaises(RuntimeError):
            self.run_engine(evaluator="tests._iso_fake_evaluator:slow")
        payload = self.run_engine(evaluator="tests._iso_fake_evaluator:slow", restart=True,
                                  timeout_s=3.0)
        self.assertNotEqual(Path(payload["path"]), self.path)
        self.assertTrue(Path(payload["path"]).is_file())

    def test_point_timeout_is_recorded_and_others_finish(self):
        import time
        started = time.perf_counter()
        payload = self.run_engine(evaluator="tests._iso_fake_evaluator:slow", timeout_s=3.0)
        self.assertLess(time.perf_counter() - started, 25.0)
        self.assertEqual(payload["status"][0, 1], "timeout")
        self.assertEqual(int(np.sum(payload["status"] == "accepted")), 10)

    def test_repair_reevaluates_only_unaccepted_points_in_place(self):
        self.run_engine(evaluator="tests._iso_fake_evaluator:slow", timeout_s=3.0)
        before = self.engine.common.load(self.path)
        payload = self.engine.repair(self.path, workers=2, evaluator="tests._iso_fake_evaluator:evaluate",
                                     timeout_s=10.0)
        self.assertEqual(payload["status"][0, 1], "accepted")
        self.assertTrue(payload["records"][0, 1]["repaired"])
        self.assertEqual(payload["identity"], before["identity"])
        self.assertEqual(payload["repairs"][-1]["points"], [(0, 1)])
        self.assertEqual(payload["new_evaluations"], 1)

    def test_worker_crash_is_recorded_and_run_continues(self):
        payload = self.run_engine(evaluator="tests._iso_fake_evaluator:crash")
        self.assertEqual(payload["status"][0, 1], "unresolved")
        self.assertIn("worker", payload["records"][0, 1]["message"])
        self.assertEqual(payload["records"][0, 1]["x"], 2.0)
        self.assertEqual(int(np.sum(payload["status"] == "accepted")), 10)

    def test_repair_can_target_named_points(self):
        self.run_engine(evaluator="tests._iso_fake_evaluator:slow", timeout_s=3.0)
        payload = self.engine.common.load(self.path)
        payload["status"][1, 2] = "unresolved"   # a second unresolved point that must stay untouched
        self.engine.common.atomic_save(self.path, payload)
        repaired = self.engine.repair(self.path, workers=2, evaluator="tests._iso_fake_evaluator:evaluate",
                                      points=[(0, 1)])
        self.assertEqual(repaired["status"][0, 1], "accepted")
        self.assertEqual(repaired["status"][1, 2], "unresolved")
        self.assertEqual(repaired["repairs"][-1]["points"], [(0, 1)])

    def test_repair_refuses_a_payload_from_another_solver(self):
        self.run_engine(evaluator="tests._iso_fake_evaluator:slow", timeout_s=3.0)
        payload = self.engine.common.load(self.path)
        payload["identity"]["sources"]["RMFsolver/phase_velocity.py"] = "another solver"
        self.engine.common.atomic_save(self.path, payload)
        with self.assertRaises(RuntimeError):
            self.engine.repair(self.path, workers=2, evaluator="tests._iso_fake_evaluator:evaluate")
        repaired = self.engine.repair(self.path, workers=2, evaluator="tests._iso_fake_evaluator:evaluate",
                                      force=True)
        self.assertEqual(repaired["status"][0, 1], "accepted")

    def test_identity_covers_eos_modules_and_library_versions(self):
        identity = self.engine.common.identity()
        for module in ("Solver.py", "SQMsolver.py", "RMFparameter.py", "constants.py", "phase_velocity.py"):
            self.assertIn(f"RMFsolver/{module}", identity["sources"])
        self.assertEqual(sorted(identity["versions"]), ["numpy", "scipy"])

    def test_seed_refuses_a_changed_eos_module(self):
        first = self.run_engine()
        seed = self.engine.common.load(first["path"])
        seed["identity"]["sources"]["RMFsolver/Solver.py"] = "another EOS"
        self.engine.common.atomic_save(first["path"], seed)
        with self.assertRaises(RuntimeError):
            self.run_engine(out_path=self.directory / "second.npy", seed_from=first["path"])

    def test_seed_from_previous_payload_reuses_matching_points(self):
        first = self.run_engine()
        mesh = synthetic_mesh()
        finer = dict(mesh, fingerprint="finer", theta=np.array([0.1, 0.45, 0.8, 1.2, np.pi / 2]))
        for key in ("x", "T", "rho"):
            finer[key] = np.insert(mesh[key], [1, 2], mesh[key][[0, 1]] + 0.123, axis=0)
        finer["point_status"] = np.where(finer["T"] >= 8.0, "pending", "outside_panel").astype(object)
        payload = self.run_engine(mesh=finer, out_path=self.directory / "finer.npy", seed_from=first["path"])
        reused = int(np.sum(first["status"] == "accepted"))
        self.assertEqual(payload["seeded"], reused)
        self.assertEqual(payload["new_evaluations"], int(np.sum(finer["point_status"] == "pending")) - reused)
        np.testing.assert_array_equal(payload["speed_m_s"][[0, 2, 4]], first["speed_m_s"])

    def test_profiles_sidecar_matches_accepted_points(self):
        payload = self.run_engine()
        files = sorted(p.name for p in Path(payload["profiles_dir"]).glob("*.npz"))
        accepted = np.argwhere(payload["status"] == "accepted")
        self.assertEqual(files, sorted(f"{i}_{j}.npz" for i, j in accepted))


def small_analytic_payload():
    """Five real analytic records on a 1x5 'mesh' (no engine needed)."""
    import _iso_common as common
    import _iso_evaluate as evaluation
    points = [(3.2, 10.0), (4.0, 10.0), (2.5, 55.0), (3.8, 40.0), (1.5, 80.0)]
    records = np.empty((1, len(points)), dtype=object)
    for j, (x, T) in enumerate(points):
        records[0, j] = evaluation.evaluate("analytic", x, T)[0]
    x = np.array([[p[0] for p in points]])
    T = np.array([[p[1] for p in points]])
    mesh = dict(x=x, T=T, rho=np.linspace(0.3, 0.7, 5)[None, :], fractions=np.linspace(.1, .9, 5),
                fingerprint="five-points")
    status = np.array([[r["status"] for r in records[0]]], dtype=object)
    return dict(identity=common.identity(method="analytic", mesh="five-points", evaluator="test"),
                mesh=mesh, records=records, status=status,
                speed_m_s=np.array([[r["speed_m_s"] for r in records[0]]]))


def synthetic_cache(xi_true=0.3):
    fractions = np.linspace(0.05, 0.95, 19)
    a = np.tile(0.1 + 0.85 * fractions, (3, 1))
    I2 = np.full_like(a, 1e-14) * (1 + np.arange(3))[:, None]
    lam = np.full_like(a, 0.6)
    rho_in, rho_out = np.array([0.3, 0.35, 0.4])[:, None], np.array([0.9, 0.85, 0.8])[:, None]
    rho = rho_in + fractions[None, :] * (rho_out - rho_in)
    return dict(I2=I2, lambda_n=lam, a_0plus=a, accepted=np.ones_like(a, dtype=bool), rho=rho,
                fractions=fractions, x=1 + rho, T=60 * rho, provenance=dict(phase_velocity_sha256="any"))


class XiCacheTests(unittest.TestCase):
    def test_retuned_grid_matches_direct_analytic(self):
        import _iso_common as common
        import _iso_xi as xi_cache
        payload = small_analytic_payload()
        cache = xi_cache.build_cache(payload)
        for xi in (-0.85, -0.4, 0.0, 0.5):
            speeds = xi_cache.retune(cache, xi)
            for j in range(5):
                direct = pv.analytic_velocity_isothermal_slow(
                    payload["mesh"]["T"][0, j], payload["mesh"]["x"][0, j] * common.N0_MEV3,
                    common.PHYSICS["B_one_forth"], xi=xi)["velocity_m_s"]
                with self.subTest(xi=xi, j=j):
                    self.assertLess(abs(speeds[0, j] / direct - 1), 1e-12)

    def test_retune_rejects_changed_source(self):
        import _iso_xi as xi_cache
        cache = synthetic_cache()
        cache["provenance"]["phase_velocity_sha256"] = "stale"
        with self.assertRaises(RuntimeError):
            xi_cache.retune(cache, 0.0, check_source=True)
        self.assertTrue(np.all(np.isfinite(xi_cache.retune(cache, 0.0, check_source=False))))

    def test_effective_xi_recovers_injected_value(self):
        import _iso_xi as xi_cache
        cache = synthetic_cache()
        numerical = dict(speed_m_s=xi_cache.retune(cache, 0.3, check_source=False))
        np.testing.assert_allclose(xi_cache.effective_xi(cache, numerical), 0.3, atol=1e-10)

    def test_fit_xi_recovers_injected_value(self):
        import _iso_xi as xi_cache
        cache = synthetic_cache()
        numerical = dict(speed_m_s=xi_cache.retune(cache, 0.3, check_source=False))
        levels = np.geomspace(np.nanmin(numerical["speed_m_s"]) * 1.5,
                              np.nanmax(numerical["speed_m_s"]) / 1.5, 5)
        for objective in ("log_speed", "contour"):
            with self.subTest(objective=objective):
                self.assertAlmostEqual(xi_cache.fit_xi(cache, numerical, levels, objective, check_source=False), 0.3, places=4)

    def test_contour_fit_on_row_mesh_recovers_injected_value(self):
        import _iso_xi as xi_cache
        cache = synthetic_cache()
        cache["rho"] = None   # row mesh: offsets along each row in (x - 1)/4.5
        cache["x"] = 2.9 + cache["fractions"][None, :] * np.array([2.3, 2.35, 2.4])[:, None]
        numerical = dict(speed_m_s=xi_cache.retune(cache, 0.3, check_source=False))
        levels = np.geomspace(np.nanmin(numerical["speed_m_s"]) * 1.5,
                              np.nanmax(numerical["speed_m_s"]) / 1.5, 5)
        offsets = xi_cache.contour_offsets(cache, numerical, 0.3, levels, check_source=False)
        self.assertGreater(np.isfinite(offsets).sum(), 10)
        np.testing.assert_allclose(offsets[np.isfinite(offsets)], 0.0, atol=1e-12)
        self.assertAlmostEqual(xi_cache.fit_xi(cache, numerical, levels, "contour", check_source=False),
                               0.3, places=4)

    def test_contour_crossings_never_bridge_failed_points(self):
        import _iso_xi as xi_cache
        fractions = np.linspace(0.0, 1.0, 5)
        speeds = np.array([[10.0, 20.0, np.nan, 80.0, 160.0]])
        self.assertTrue(np.isnan(xi_cache._ray_crossings(speeds, fractions[None, :], fractions, 40.0)[0]))
        self.assertAlmostEqual(xi_cache._ray_crossings(speeds, fractions[None, :], fractions,
                                                       np.sqrt(80.0 * 160.0))[0], 0.875)

    def test_fit_xi_refuses_numerical_data_from_another_mesh(self):
        import _iso_xi as xi_cache
        cache = synthetic_cache()
        cache["provenance"]["mesh_fingerprint"] = "mesh A"
        numerical = dict(speed_m_s=xi_cache.retune(cache, 0.3, check_source=False),
                         mesh=dict(fingerprint="mesh B"))
        with self.assertRaises(ValueError):
            xi_cache.fit_xi(cache, numerical, (), "log_speed", check_source=False)
        with self.assertRaises(ValueError):
            xi_cache.effective_xi(cache, numerical)

    def test_xi_module_imports_no_solver(self):
        import subprocess
        code = ("import sys; sys.path.insert(0, %r); import _iso_xi; "
                "assert 'RMFsolver.phase_velocity' not in sys.modules") % str(WORKFLOW)
        subprocess.run([sys.executable, "-c", code], check=True)


def fake_payload(speed=None):
    mesh = synthetic_mesh()
    status = mesh["point_status"].copy()
    speeds = 50.0 * np.exp(mesh["x"]) * (1 + mesh["T"] / 100.0) if speed is None else speed
    status[status == "pending"] = "accepted"
    speeds = np.where(status == "accepted", speeds, np.nan)
    return dict(mesh=mesh, status=status, speed_m_s=speeds)


class PlotTests(unittest.TestCase):
    def test_outside_panel_points_masked_in_contour_arrays(self):
        import _iso_plot as plot
        x, T, logv = plot.contour_arrays(fake_payload())
        outside = synthetic_mesh()["point_status"] == "outside_panel"
        self.assertTrue(np.all(logv.mask[outside]))
        self.assertFalse(np.any(logv.mask[~outside]))

    def ring_payload(self, unresolved=None):
        import _iso_common as common
        theta = np.array([0.4, 0.8, 1.2])
        fractions = np.array([0.005, 0.1, 0.5, 0.9, 0.99, 0.999])
        inner, outer = np.array([0.4, 0.42, 0.45]), np.array([0.9, 0.85, 0.8])
        rho = inner[:, None] + fractions[None, :] * (outer - inner)[:, None]
        x, T = common.to_physical(rho, np.broadcast_to(theta[:, None], rho.shape))
        u = -np.log1p(-fractions)
        speed = np.broadcast_to(10 ** (2 + 0.5 * u), rho.shape).copy()
        status = np.full(rho.shape, "accepted", dtype=object)
        if unresolved:
            status[unresolved], speed[unresolved] = "unresolved", np.nan
        boundaries = [dict(rho_inner=a, rho_outer=b) for a, b in zip(inner, outer)]
        mesh = dict(theta=theta, fractions=fractions, rho=rho, x=np.asarray(x), T=np.asarray(T),
                    point_status=status.copy(), boundaries=boundaries, fingerprint="ring")
        return dict(mesh=mesh, status=status, speed_m_s=speed)

    def test_radial_upsampling_is_exact_for_linear_logv_in_u(self):
        import _iso_common as common
        import _iso_plot as plot
        x, T, logv = plot.contour_arrays(self.ring_payload(), radial_upsample=4)
        self.assertEqual(logv.shape, (3, 21))
        rho, theta = common.to_polar(x, T)
        fraction = (rho - np.array([0.4, 0.42, 0.45])[:, None]) / np.array([0.5, 0.43, 0.35])[:, None]
        np.testing.assert_allclose(logv, 2 + 0.5 * -np.log1p(-fraction), atol=1e-10)

    def test_radial_upsampling_keeps_failures_as_holes(self):
        import _iso_plot as plot
        x, T, logv = plot.contour_arrays(self.ring_payload(unresolved=(1, 3)), radial_upsample=4)
        self.assertTrue(np.all(logv.mask[1, 9:16]))     # between coarse columns 2 and 4
        self.assertFalse(np.any(logv.mask[1, :9]))
        self.assertFalse(np.any(logv.mask[[0, 2]]))

    def test_coverage_reports_rays_missing_top_level(self):
        import _iso_plot as plot
        payload = fake_payload()
        report = plot.coverage(payload, levels=(300.0, 1e6))
        self.assertEqual(report[1e6]["missing"], {0: "above_outer_column", 1: "above_outer_column",
                                                  2: "above_outer_column"})
        self.assertIn(1, report[300.0]["bracketed"])

    def test_gray_polygons_lie_outside_mesh_points(self):
        from matplotlib.path import Path as MplPath
        import _iso_boundaries as boundaries
        import _iso_plot as plot
        import polar_mesh
        mesh = polar_mesh.build_mesh(workers=4, theta=np.radians((15.0, 45.0, 75.0)))
        curves = boundaries.boundary_curves(n_rows=4, n_rays=6, workers=4, extra_rays=mesh["boundaries"])
        polygons = plot.gray_polygons(curves, x_limits=(1.0, 5.5), T_limits=(0.0, 115.0))
        points = np.column_stack([mesh["x"][mesh["point_status"] == "pending"],
                                  mesh["T"][mesh["point_status"] == "pending"]])
        for name, polygon in polygons.items():
            with self.subTest(name=name):
                self.assertFalse(np.any(MplPath(polygon).contains_points(points)))

    def test_all_plots_new_isothermal_section_uses_the_reference_styles(self):
        import ast
        import nbformat
        import _iso_plot as plot
        cells = [c for c in nbformat.read(ROOT / "All_Plots_new.ipynb", as_version=4).cells
                 if "isothermal-contours-26-10-27" in c.metadata.get("tags", [])]
        constants = {}
        for cell in cells:
            if cell.cell_type == "code":
                for node in ast.parse(cell.source).body:
                    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                        try:
                            constants[node.targets[0].id] = ast.literal_eval(node.value)
                        except ValueError:
                            pass
        import update_all_plots_new as generator
        expected = generator.manuscript_style(plot.LINEAR_REFERENCE_STYLE)   # reference plus manuscript fonts
        for key, value in expected.items():
            with self.subTest(key=key):
                self.assertEqual(constants["ISO_LIN_" + key.upper()], value)
        self.assertGreaterEqual(plot.LINEAR_REFERENCE_STYLE["contour_label_font_size"], 16)
        latex = "Latex_writting/Hydrodynamical combustion/Plots/"
        self.assertEqual(constants["ISO_LIN_OUTPUT_LATEX_PDF"], latex + "isothermal-numerical-contour.pdf")
        self.assertEqual(constants["ISO_LOGN_OUTPUT_LATEX_PDF"], latex + "isothermal-numerical-contour-logT.pdf")
        self.assertEqual(constants["ISO_CMP_OUTPUT_LATEX_PDF"], latex + "isothermal-analytic-numerical-logT.pdf")
        self.assertFalse(constants["ISO_LOGN_SHOW_METHOD_LEGEND"])
        width, height = constants["ISO_LOGN_FIGURE_SIZE"]
        self.assertLess(height, width)   # requested landscape panel
        self.assertLess(constants["ISO_LOGN_BOX_ASPECT"], 1.0)   # the axes box, not only the canvas
        self.assertLessEqual(width, 8.0)   # 16/18 pt font floor applies up to 8 in
        numerical_log = next(c.source for c in cells if "ISO_LOGN_TITLE" in c.source)
        self.assertNotIn("retune", numerical_log)   # numerical only

    def test_all_plots_new_isothermal_fonts_print_at_least_at_body_text_size(self):
        import ast
        import nbformat
        cells = [c for c in nbformat.read(ROOT / "All_Plots_new.ipynb", as_version=4).cells
                 if c.cell_type == "code" and "isothermal-contours-26-10-27" in c.metadata.get("tags", [])]
        column_pt, body_pt = 246.0, 10.0   # revtex4-1 10pt twocolumn \linewidth and body text
        checked = 0
        for cell in cells:
            constants = {}
            for node in ast.parse(cell.source).body:
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                    try:
                        constants[node.targets[0].id] = ast.literal_eval(node.value)
                    except ValueError:
                        pass
            sizes = [k for k in constants if k.endswith("FIGURE_SIZE")]
            if not sizes:
                continue
            width_in = constants[sizes[0]][0]
            for name, value in constants.items():
                if name.endswith("FONT_SIZE"):
                    checked += 1
                    with self.subTest(font=name):
                        self.assertGreaterEqual(value * column_pt / (72.0 * width_in), body_pt)
        self.assertGreaterEqual(checked, 20)

    def test_all_plots_new_writes_the_fitted_xi_comparison(self):
        import ast
        import json
        import nbformat
        cells = [c for c in nbformat.read(ROOT / "All_Plots_new.ipynb", as_version=4).cells
                 if "isothermal-contours-26-10-27" in c.metadata.get("tags", [])]
        source = next(c.source for c in cells if c.cell_type == "code" and "ISO_FIT_XI" in c.source)
        constants = {}
        for node in ast.parse(source).body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                try:
                    constants[node.targets[0].id] = ast.literal_eval(node.value)
                except ValueError:
                    pass
        self.assertEqual(constants["ISO_FIT_OUTPUT_LATEX_PDF"], "Latex_writting/Hydrodynamical combustion/Plots/"
                                                                 "isothermal-analytical-numerical-comparison.pdf")
        fit = json.loads((WORKFLOW / "xi_fit/xi_fit.json").read_text())["fits"]["lowT"]["contour"]["xi"]
        self.assertEqual(constants["ISO_FIT_XI"], round(fit, 2))
        self.assertTrue(constants["ISO_FIT_SHOW_METHOD_LEGEND"])

    def test_no_contour_title_in_all_plots_new_uses_u_0minus(self):
        import ast
        import nbformat
        titles = {}
        for index, cell in enumerate(nbformat.read(ROOT / "All_Plots_new.ipynb", as_version=4).cells):
            if cell.cell_type != "code":
                continue
            for node in ast.parse(cell.source).body:
                if (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                        and node.targets[0].id.endswith("TITLE") and "LEGEND" not in node.targets[0].id):
                    titles[(index, node.targets[0].id)] = ast.literal_eval(node.value)
        contour_titles = {k: v for k, v in titles.items() if "[" in v and "m" in v and "s^{-1}" in v}
        self.assertGreaterEqual(len(contour_titles), 5)
        for key, title in contour_titles.items():
            with self.subTest(cell=key):
                self.assertNotIn("u(0^-)", title)
                self.assertIn("speed", title)


class LowTTests(unittest.TestCase):
    def test_row_mesh_is_strictly_inside_each_row_window(self):
        sys.path.insert(0, str(WORKFLOW / "lowT"))
        import row_mesh
        mesh = row_mesh.build_mesh()
        self.assertEqual(mesh["kind"], "rows")
        self.assertEqual(mesh["x"].shape, (40, len(row_mesh.FRACTIONS)))
        np.testing.assert_allclose(mesh["T_rows"], np.geomspace(0.01, 10.0, 40))
        self.assertTrue(np.all(mesh["x"] > mesh["x_inner"][:, None]))
        self.assertTrue(np.all(mesh["x"] < mesh["x_outer"][:, None]))
        self.assertTrue(np.all(mesh["point_status"] == "pending"))
        self.assertTrue(np.all((2.8 < mesh["x_inner"]) & (mesh["x_inner"] < 3.0)))
        self.assertTrue(np.all((5.2 < mesh["x_outer"]) & (mesh["x_outer"] < 5.35)))

    def test_stale_boundary_curves_are_refused(self):
        import tempfile
        sys.path.insert(0, str(WORKFLOW / "lowT"))
        import build_boundary_curves
        import row_mesh
        stale = Path(tempfile.mkdtemp()) / "boundary-curves.npz"
        data = dict(np.load(build_boundary_curves.OUTPUT))
        np.savez(stale, **dict(data, key="made with other sources"))
        with patch.object(build_boundary_curves, "OUTPUT", stale):
            with self.assertRaises(RuntimeError):
                row_mesh.build_mesh()
            with self.assertRaises(RuntimeError):
                build_boundary_curves.load()

    def rows_payload(self):
        fractions = np.array([1e-4, 1e-3, 0.1, 0.5, 0.9, 0.999])
        T_rows = np.array([0.01, 0.1, 1.0])
        inner, outer = np.array([2.99, 2.98, 2.95]), np.array([5.30, 5.29, 5.27])
        x = inner[:, None] + fractions[None, :] * (outer - inner)[:, None]
        logit = np.log(fractions / (1 - fractions))
        speed = np.broadcast_to(10 ** (3 + 0.25 * logit), x.shape).copy()
        mesh = dict(kind="rows", T_rows=T_rows, x_inner=inner, x_outer=outer, fractions=fractions, x=x,
                    T=np.broadcast_to(T_rows[:, None], x.shape).copy(), point_status=np.full(x.shape, "pending", dtype=object),
                    fingerprint="rows")
        return dict(mesh=mesh, status=np.full(x.shape, "accepted", dtype=object), speed_m_s=speed)

    def test_logit_upsampling_is_exact_for_linear_logv_in_logit(self):
        import _iso_plot as plot
        payload = self.rows_payload()
        x, T, logv = plot.contour_arrays(payload, radial_upsample=4, radial_variable="logit")
        f = (x - payload["mesh"]["x_inner"][:, None]) / (payload["mesh"]["x_outer"] - payload["mesh"]["x_inner"])[:, None]
        np.testing.assert_allclose(logv, 3 + 0.25 * np.log(f / (1 - f)), atol=1e-10)
        np.testing.assert_array_equal(T[:, 0], payload["mesh"]["T_rows"])

    def test_contour_lines_in_log_T_follow_the_level(self):
        import _iso_plot as plot
        x, T, logv = plot.contour_arrays(self.rows_payload(), radial_upsample=4, radial_variable="logit")
        lines = plot.contour_lines(x, T, logv, levels=(1000.0,), log_T=True)
        vertices = np.vstack(lines[1000.0])
        self.assertTrue(np.all((vertices[:, 1] >= 0.01 - 1e-12) & (vertices[:, 1] <= 1.0 + 1e-12)))
        expected = np.interp(vertices[:, 1], [0.01, 0.1, 1.0], [2.99 + 0.5 * 2.31, 2.98 + 0.5 * 2.31, 2.95 + 0.5 * 2.32])
        np.testing.assert_allclose(vertices[:, 0], expected, atol=0.02)

    def test_log_style_matches_reference_values(self):
        import ast
        import subprocess
        import _iso_plot as plot
        source = subprocess.run([sys.executable, str(ROOT / ".claude/tools/nbsrc.py"),
                                 str(ROOT / "26-10-02.ipynb"), "-c", "8"],
                                capture_output=True, text=True, check=True).stdout
        reference = {}
        for line in source.splitlines():
            if " = " in line and line.split(" = ")[0].isupper():
                name, value = line.split(" = ", 1)
                try:
                    reference[name.lower()] = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    pass
        style = plot.LOG_REFERENCE_STYLE
        compared = 0
        for key, value in reference.items():
            if key in plot.LOG_STYLE_OVERRIDES or key not in style:
                continue
            compared += 1
            with self.subTest(key=key):
                self.assertEqual(style[key], value)
        self.assertGreaterEqual(compared, 40)
        self.assertEqual(style["y_limits"], (0.01, 10.0))
        for key in ("legend_font_size", "speed_legend_font_size", "contour_label_font_size"):
            self.assertGreaterEqual(style[key], 16)

    def test_log_figure_numerical_only_has_no_dashes_or_method_legend(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import _iso_plot as plot
        style = dict(plot.LOG_REFERENCE_STYLE, show_method_legend=False)
        logT = np.linspace(-2, 1, 50)
        lines = {level: [np.column_stack([3.0 + 0.25 * k - 0.1 * logT, 10 ** logT])]
                 for k, level in enumerate(style["contour_levels"])}
        curves = dict(inner=(np.full(50, 2.95), 10 ** logT), outer=(np.full(50, 5.3), 10 ** logT))
        figure, axis = plt.subplots()
        plot.draw_log_reference(axis, curves, lines, {}, None, style)
        self.assertEqual({line.get_linestyle() for line in axis.get_lines()}, {"-"})
        self.assertEqual([c for c in axis.get_children() if isinstance(c, matplotlib.legend.Legend)], [])
        self.assertTrue(set(style["speed_labels"]) <= {t.get_text() for t in axis.texts})
        plt.close(figure)

    def test_log_figure_labels_contours_without_speed_legend(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import _iso_plot as plot
        style = plot.LOG_REFERENCE_STYLE
        self.assertEqual(style["contour_levels"], (1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6))
        self.assertEqual(len(set(style["contour_colors"])), 9)
        self.assertGreaterEqual(style["x_limits"][1], 5.75)   # room for the udd strip label
        self.assertFalse(style["show_speed_legend"])
        self.assertEqual(style["inner_region_text"], plot.LINEAR_REFERENCE_STYLE["inner_region_text"])
        self.assertIn("$udd$ quark", style["outer_region_text"])
        self.assertIn("favored", style["outer_region_text"])
        logT = np.linspace(-2, 1, 50)
        lines = {level: [np.column_stack([3.0 + 0.5 * k - 0.1 * logT, 10 ** logT])]
                 for k, level in enumerate(style["contour_levels"])}
        curves = dict(inner=(np.full(50, 2.95), 10 ** logT), outer=(np.full(50, 5.3), 10 ** logT))
        figure, axis = plt.subplots()
        plot.draw_log_reference(axis, curves, lines, lines, -0.85, style)
        texts = {t.get_text(): t for t in axis.texts}
        for level, label in zip(style["contour_levels"], style["speed_labels"]):
            with self.subTest(level=level):
                self.assertIn(label, texts)
                x, T = texts[label].get_position()
                self.assertAlmostEqual(T, style["label_temperatures"][level])
                self.assertAlmostEqual(x, 3.0 + 0.5 * style["contour_levels"].index(level) - 0.1 * np.log10(T))
                self.assertLess(abs(texts[label].get_rotation() - 90) % 180, 30)   # along a near-vertical line
        legends = [child for child in axis.get_children() if isinstance(child, matplotlib.legend.Legend)]
        self.assertEqual(len(legends), 1)   # the method legend only
        plt.close(figure)


if __name__ == "__main__":
    unittest.main()
