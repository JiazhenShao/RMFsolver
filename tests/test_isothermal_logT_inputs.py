"""Portable, provenance-checked warm-input snapshot tests (no EOS)."""

import importlib.util
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


MODULE = Path(__file__).resolve().parents[1] / "new_paper_calculations/26-10-01/isothermal_numeric_logT/_logT_inputs.py"
sys.path.insert(0, str(MODULE.parent))
SPEC = importlib.util.spec_from_file_location("logT_inputs", MODULE)
inputs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inputs)


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, payload, allow_pickle=True)


class InputsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "project"
        self.directory = Path(self.temp.name) / "run"
        source = self.root / "local_analysis_results/All_Plots_new/isothermal_contour_data"
        source.mkdir(parents=True)
        n0 = 1_229_360.5098323745
        self.n0 = n0
        save(source / "isothermal-domain.npy", dict(payload_kind="isothermal_domain",
            domain_fingerprint="physics-A", nB_0minus_grid=np.array([[3*n0, 4*n0]]),
            nB_0minus_over_n0_grid=np.array([[3., 4.]]),
            B_one_forth_MeV=189.1565957288247, ms_MeV=0., NM_type="PNM", upB=5000, xi=-.5))
        good = dict(success=True, T_0minus=.5, nB_0minus=3*n0,
                    velocity_m_s=75., jB=1., bvp_status=0)
        bad = dict(good, success=False)
        save(source / "isothermal-numerical.npy", dict(payload_kind="isothermal_numerical",
            domain_fingerprint="physics-A", task_status=np.array([["success", "failed"]], object),
            B_one_forth_MeV=189.1565957288247, ms_MeV=0., NM_type="PNM", upB=5000, xi=-.5,
            T_0minus_grid_MeV=np.array([[.5, .5]]),
            nB_0minus_grid=np.array([[3*n0, 4*n0]]),
            velocity_m_s_grid=np.array([[75., 999.]]), jB_grid=np.array([[1., 99.]]),
            records=np.array([[good, bad]], object)))
        base_hashes = {name: hashlib.sha256((source/name).read_bytes()).hexdigest()
                       for name in ("isothermal-domain.npy", "isothermal-numerical.npy")}
        accepted = dict(nB_0minus_over_n0=3.01, nB_0minus=3.01*n0,
                        T_0minus=.4, velocity_m_s=80., jB=1.2,
                        result=dict(success=True, bvp_status=0))
        rejected = dict(accepted, nB_0minus_over_n0=3.02,
                        result=dict(success=False, bvp_status=1))
        save(source / "isothermal-contour-repair.npy", dict(payload_kind="isothermal_contour_repair",
            domain_fingerprint="physics-A", baseline_hashes=base_hashes,
            points={"accepted": accepted, "failed": rejected}))
        save(source / "isothermal-cold-endpoints.npy", dict(payload_kind="isothermal_cold_endpoints",
            domain_fingerprint="physics-A", baseline_hashes=base_hashes,
            points={"cold": dict(accepted,
                T_0minus=.1, nB_0minus_over_n0=3.05, nB_0minus=3.05*n0,
                velocity_m_s=100., jB=1.5)}))
        np.savez(source / "isothermal-numerical-contour.npz",
                 baseline_density_grid=np.array([[3., 4.]]),
                 baseline_temperature_grid=np.array([[.5, .5]]),
                 baseline_success_grid=np.array([[True, False]]),
                 curve_vertices_0=np.array([[3., .5], [3.5, .45]]))

    def test_bundle_loads_only_accepted_physical_points_and_warm_arrays(self):
        inputs.bundle_inputs(self.root, self.directory)
        result = inputs.load_inputs(self.directory)
        self.assertEqual(result["fingerprint"], "physics-A")
        self.assertAlmostEqual(result["n0"], self.n0)
        self.assertEqual(len(result["points"]), 3)
        self.assertEqual({p["velocity_m_s"] for p in result["points"]}, {75., 80., 100.})
        self.assertEqual({p["source"] for p in result["points"]}, {
            "snapshot:inputs/isothermal-numerical.npy",
            "snapshot:inputs/isothermal-contour-repair.npy",
            "snapshot:inputs/isothermal-cold-endpoints.npy",
        })
        for p in result["points"]:
            self.assertEqual(p["id"], f'{float(p["T_0minus"]).hex()}:{float(p["nB_0minus_over_n0"]).hex()}')
        self.assertIn("baseline_density_grid", result["warm"])
        self.assertEqual(result["warm"]["curve_vertices_0"].shape, (2, 2))
        self.assertEqual(len(result["manifest"]["files"]), 5)
        self.assertEqual(result["manifest"]["physics"]["NM_type"], "PNM")

    def test_snapshot_is_immutable_and_hash_checked(self):
        inputs.bundle_inputs(self.root, self.directory)
        copied = self.directory / "inputs/isothermal-domain.npy"
        original = copied.read_bytes()
        source = self.root / "local_analysis_results/All_Plots_new/isothermal_contour_data/isothermal-domain.npy"
        source.write_bytes(b"new source bytes")
        inputs.bundle_inputs(self.root, self.directory)
        self.assertEqual(copied.read_bytes(), original)
        copied.write_bytes(b"corrupted")
        with self.assertRaisesRegex(ValueError, "hash"):
            inputs.load_inputs(self.directory)

    def test_optional_probe_profile_is_independent_and_usable(self):
        probe = self.root / "local_analysis_results/isothermal_low_temperature_probe/core_mesh_T0.01_n3.001.npy"
        state = dict(s=np.array([0., .5]), nK=np.array([2., 1.]),
                     jK=np.array([1., .5]), nK_inf=0., jK_inf=0., jB=1.)
        save(probe, dict(T_0minus=.01, nB_0minus_over_n0=3.001,
                         B_one_forth=189.1565957288247,
                         physics=dict(ms=0., NM_type="PNM", upB=5000, xi=-.5),
                         result=dict(success=True, bvp_status=0, velocity_m_s=74.,
                                     jB=1., nB_0minus=3.001*self.n0,
                                     continuation_state=state,
                                     endpoint_flux_residual_max=1e-12)))
        inputs.bundle_inputs(self.root, self.directory)
        result = inputs.load_inputs(self.directory)
        p = next(p for p in result["points"] if p["T_0minus"] == .01)
        self.assertTrue(p["profile_file"].startswith("inputs/profiles/"))
        self.assertEqual(p["source"], "snapshot:" + p["profile_file"])
        self.assertEqual(np.load(self.directory/p["profile_file"], allow_pickle=True).item()["jB"], 1.)
        self.assertIn(p["profile_file"], result["manifest"]["files"])

    def test_incompatible_warm_physics_is_rejected(self):
        source = self.root / "local_analysis_results/All_Plots_new/isothermal_contour_data"
        for name, field, value in (
            ("isothermal-domain.npy", "B_one_forth_MeV", 190.),
            ("isothermal-domain.npy", "ms_MeV", 1.),
            ("isothermal-numerical.npy", "NM_type", "SYM"),
        ):
            with self.subTest(field=field):
                path = source / name
                original = np.load(path, allow_pickle=True).item()
                save(path, dict(original, **{field: value}))
                with self.assertRaisesRegex(ValueError, "physics"):
                    inputs.bundle_inputs(self.root, self.directory / field)
                save(path, original)

    def test_incompatible_probe_is_not_imported(self):
        probe = self.root / "local_analysis_results/isothermal_low_temperature_probe/core_mesh_T0.01_n3.001.npy"
        valid = dict(T_0minus=.01, nB_0minus_over_n0=3.001,
                     B_one_forth=189.1565957288247,
                     physics=dict(ms=0., NM_type="PNM", upB=5000, xi=-.5),
                     result=dict(success=True, bvp_status=0, velocity_m_s=74., jB=1.,
                                 nB_0minus=3.001*self.n0, endpoint_flux_residual_max=1e-12,
                                 continuation_state=dict(s=np.array([0., .5]),
                                                         nK=np.array([2., 1.]),
                                                         jK=np.array([1., .5]), jB=1.)))
        for field, value in (("B_one_forth", 190.), ("ms", 1.), ("NM_type", "SYM")):
            with self.subTest(field=field):
                probe_data = dict(valid, physics=dict(valid["physics"]))
                if field == "B_one_forth":
                    probe_data[field] = value
                else:
                    probe_data["physics"][field] = value
                save(probe, probe_data)
                result = inputs.bundle_inputs(self.root, self.directory / field)
                self.assertEqual(result["manifest"]["profiles"], [])

    def test_near_current_bound_history_is_not_a_targeting_point(self):
        source = self.root / "local_analysis_results/All_Plots_new/isothermal_contour_data/isothermal-contour-repair.npy"
        payload = np.load(source, allow_pickle=True).item()
        payload["points"]["accepted"]["current_bracket_near_bound"] = True
        save(source, payload)
        result = inputs.bundle_inputs(self.root, self.directory)
        self.assertEqual({p["velocity_m_s"] for p in result["points"]}, {75., 100.})

    def test_repair_source_must_match_validated_baseline_hashes(self):
        source = self.root / "local_analysis_results/All_Plots_new/isothermal_contour_data/isothermal-contour-repair.npy"
        payload = np.load(source, allow_pickle=True).item()
        payload["baseline_hashes"]["isothermal-domain.npy"] = "bad"
        save(source, payload)
        with self.assertRaisesRegex(ValueError, "physics|baseline"):
            inputs.bundle_inputs(self.root, self.directory)

    def test_optional_historical_cache_is_copied_only_when_matching_point(self):
        source = self.root / "local_analysis_results/All_Plots_new/isothermal_contour_data/isothermal-cold-endpoints.npy"
        cache = self.root / "new_paper_calculations/26-09-23/isothermal_cold_endpoints/isothermal-cold-endpoints-profiles/known.npy"
        payload = np.load(source, allow_pickle=True).item()
        payload["points"]["cold"]["profile_file"] = "isothermal-cold-endpoints-profiles/known.npy"
        save(source, payload)
        profile = dict(s=np.array([0., .5]), nK=np.array([2., 1.]),
                       jK=np.array([1., .5]), jB=1.5, T_0minus=.1,
                       nB_0minus=3.05*self.n0)
        save(cache, profile)
        loaded = inputs.bundle_inputs(self.root, self.directory)
        cold = next(p for p in loaded["points"] if p["T_0minus"] == .1)
        self.assertTrue(cold["profile_file"].startswith("inputs/profiles/"))
        self.assertEqual(np.load(self.directory/cold["profile_file"], allow_pickle=True).item()["jB"], 1.5)
        self.assertIn("source_sha256", loaded["manifest"]["files"][cold["profile_file"]])
        profile["jB"] = 2.0
        save(cache, profile)
        other = inputs.bundle_inputs(self.root, self.directory / "mismatch")
        self.assertFalse(any(p.get("profile_file") for p in other["points"]))
