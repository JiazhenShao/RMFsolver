"""Bounded orchestration checks only; never invokes the live EOS/BVP."""
import copy
import importlib
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "new_paper_calculations/26-09-21/isothermal_targeted"
OLD = ROOT / "new_paper_calculations/26-09-16/isothermal_numeric_repair"
sys.path.insert(0, str(WORK))
try:
    targeted = importlib.import_module("_targeted")
except ModuleNotFoundError:
    targeted = None


class TargetedTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(targeted, "targeted workflow is not implemented")

    def test_real_manifest_has_twenty_named_targets(self):
        repair = np.load(OLD / "isothermal-contour-repair.npy", allow_pickle=True).item()
        tasks = targeted.initial_targets(targeted.source_roots(repair))
        self.assertEqual(len(tasks), 20)
        self.assertEqual([sum(t["phase"] == p for t in tasks) for p in targeted.PHASES], [8, 7, 5])
        self.assertTrue(all(t["T_0minus"] >= 1 for t in tasks))
        self.assertFalse(any(t["level"] == 5000 for t in tasks))

    def test_more_source_roots_do_not_expand_initial_manifest(self):
        repair=np.load(OLD/"isothermal-contour-repair.npy",allow_pickle=True).item()
        roots=targeted.source_roots(repair)
        a=next(p for p in roots if p["level"]==500 and 85<p["T_0minus"]<86)
        roots.append(dict(a,T_0minus=87.,theta_rad=np.radians(77.5)))
        roots.sort(key=lambda p:(p["level"],p["T_0minus"]))
        self.assertEqual(len(targeted.initial_targets(roots)),20)

    def point(self, temperature, density, velocity=500.):
        return dict(T_0minus=temperature, nB_0minus_over_n0=density,
                    nB_0minus=density*1.e6, velocity_m_s=velocity,
                    jB=density*1.e6*velocity/299792458.)

    def test_interval_requires_speed_and_curvature(self):
        a,b = self.point(20., 4.), self.point(30., 4.1)
        middle = self.point(25., 4.05)
        self.assertTrue(targeted.certify_interval(a, middle, b, 500.)[0])
        self.assertFalse(targeted.certify_interval(a, dict(middle, nB_0minus_over_n0=4.06), b, 500.)[0])
        self.assertFalse(targeted.certify_interval(a, dict(middle, velocity_m_s=505.), b, 500.)[0])

    def engine(self, directory, **kwargs):
        return targeted.TargetRun(directory, domain={"test": True}, numerical={}, repair={"points": {}, "rays": {}},
            roots=[], identity={"test": 1}, n0=1.e6, physics={}, **kwargs)

    def test_budget_reservations_persist_across_restart(self):
        with tempfile.TemporaryDirectory() as d:
            engine = self.engine(d)
            for i in range(4):
                self.assertIsNotNone(engine.reserve("one", "warm", self.point(25,4), "solve"))
            self.assertIsNone(engine.reserve("one", "warm", self.point(25,4), "solve"))
            other = self.engine(d)
            self.assertEqual(len(other.payload["evaluations"]), 4)
            self.assertIsNone(other.reserve("one", "warm", self.point(25,4), "solve"))

    def test_global_budget_never_exceeds_120(self):
        with tempfile.TemporaryDirectory() as d:
            engine = self.engine(d)
            for phase, limit in targeted.PHASE_BUDGETS.items():
                for i in range(limit):
                    self.assertIsNotNone(engine.reserve(f"{phase}-{i}", phase, self.point(25,4), "solve"))
                self.assertIsNone(engine.reserve(f"{phase}-extra", phase, self.point(25,4), "solve"))
            for i in range(4):
                self.assertIsNotNone(engine.reserve(f"check-{i}", "validation", self.point(25,4), "check"))
            self.assertEqual(len(engine.payload["evaluations"]), 120)
            self.assertIsNone(engine.reserve("overflow", "validation", self.point(25,4), "check"))

    def test_concurrent_reservations_obey_phase_cap(self):
        from concurrent.futures import ThreadPoolExecutor
        with tempfile.TemporaryDirectory() as d:
            engine=self.engine(d)
            with ThreadPoolExecutor(max_workers=8) as pool:
                tokens=list(pool.map(lambda i:engine.reserve(f"w-{i}","warm",self.point(25,4),"solve"),range(60)))
            self.assertEqual(sum(t is not None for t in tokens),40)
            self.assertEqual(len(targeted.load(engine.output)["evaluations"]),40)

    def test_resume_rejects_identity_change(self):
        with tempfile.TemporaryDirectory() as d:
            self.engine(d)
            with self.assertRaisesRegex(ValueError, "identity"):
                targeted.TargetRun(d, domain={}, numerical={}, repair={}, roots=[], identity={"test": 2}, n0=1, physics={})

    def test_prediction_uses_temperature_not_polar_ray(self):
        roots = [dict(self.point(20,4.), level=500.), dict(self.point(30,4.1), level=500.)]
        task = targeted.make_target("bend", 500., 25., roots)
        self.assertAlmostEqual(task["predicted_density"], 4.05)
        self.assertEqual(task["left"]["T_0minus"], 20.)
        self.assertEqual(task["right"]["T_0minus"], 30.)

    def prepared(self, directory, velocity):
        def trial(**kwargs):
            p=kwargs["point"]
            v=velocity(p["nB_0minus_over_n0"])
            return dict(success=True,status="success",velocity_m_s=v,jB=p["nB_0minus"]*v/targeted.C,
                        continuation_state={"s":np.array([0.,.9]),"nK":np.array([1.,0.]),"jK":np.array([1.,0.])})
        engine=self.engine(directory,trial=trial)
        seed=dict(self.point(25,4.05),id="seed",profile_file="seed.npy",profile_origin="targeted")
        targeted.atomic_save(Path(directory)/"seed.npy", {"seed": True})
        engine.payload["points"]["seed"]=seed
        roots=[dict(self.point(20,4.),level=500.),dict(self.point(30,4.1),level=500.)]
        engine.payload["source_roots"]=roots
        task=targeted.make_target("bend",500.,25.,roots)
        engine.payload["targets"][task["id"]]=task
        engine.save()
        return engine,task

    def test_one_shot_accepts_prediction_without_extra_calls(self):
        with tempfile.TemporaryDirectory() as d:
            engine,task=self.prepared(d,lambda density:500.)
            result=engine.solve_target(task)
            self.assertTrue(result["certified"])
            self.assertEqual(len(engine.payload["evaluations"]),1)
            engine.solve_target(task)
            self.assertEqual(len(engine.payload["evaluations"]),1)

    def test_existing_root_at_target_temperature_is_not_recomputed(self):
        with tempfile.TemporaryDirectory() as d:
            engine,task=self.prepared(d,lambda density:self.fail("Do not reshoot saved roots"))
            engine.payload["source_roots"].append(dict(self.point(25,4.05),level=500.))
            result=engine.solve_target(task)
            self.assertTrue(result["certified"])
            self.assertEqual(len(engine.payload["evaluations"]),0)
            engine.solve_target(task)
            self.assertEqual(len(engine.payload["evaluations"]),0)

    def test_local_secant_corrects_density_and_flags_curvature(self):
        with tempfile.TemporaryDirectory() as d:
            engine,task=self.prepared(d,lambda density:500.+1000.*(density-4.09))
            result=engine.solve_target(task)
            self.assertEqual(result["status"],"needs_refinement")
            self.assertAlmostEqual(result["root"]["nB_0minus_over_n0"],4.09)
            self.assertLessEqual(len(engine.payload["evaluations"]),4)
            engine.add_children(task)
            self.assertEqual(len(task["children"]),2)

    def test_failed_trials_do_not_spawn_bridge_solves(self):
        with tempfile.TemporaryDirectory() as d:
            engine,task=self.prepared(d,lambda density:500.)
            engine.trial=lambda **kwargs:dict(success=False,status="trial_timeout")
            result=engine.solve_target(task)
            self.assertEqual(result["status"],"failed")
            self.assertLessEqual(len(engine.payload["evaluations"]),4)
            self.assertEqual(set(e["purpose"] for e in engine.payload["evaluations"]),{"solve"})
            engine.add_children(task)
            self.assertFalse(task["children"])

    def test_missing_profiles_causes_zero_calls(self):
        with tempfile.TemporaryDirectory() as d:
            engine,task=self.prepared(d,lambda density:500.)
            engine.payload["points"].clear()
            result=engine.solve_target(task)
            self.assertEqual(result["message"],"missing_profile")
            self.assertEqual(len(engine.payload["evaluations"]),0)

    def test_cold_connection_checks_the_1_mev_floor(self):
        with tempfile.TemporaryDirectory() as d:
            engine=self.engine(d)
            engine.payload["source_roots"]=[dict(self.point(t,3+t/100),level=500.) for t in (.8,1.3,2.)]
            engine.seed_cold_connections()
            self.assertIn(1.,[t["T_0minus"] for t in engine.payload["targets"].values()])

    def test_saved_evaluation_callback_runs_after_atomic_write(self):
        with tempfile.TemporaryDirectory() as d:
            engine,task=self.prepared(d,lambda density:500.)
            observed=[]
            def callback(event,total):
                disk=targeted.load(engine.output)
                observed.append((len(disk["evaluations"]),disk["evaluations"][-1]["status"]))
            engine.evaluation_saved=callback
            engine.solve_target(task)
            self.assertEqual(observed,[(1,"success")])

    def test_accuracy_check_counts_in_budget_and_preserves_profile(self):
        with tempfile.TemporaryDirectory() as d:
            engine,task=self.prepared(d,lambda density:500.)
            engine.solve_target(task)
            profile=engine.profile_path(task["root"])
            before=profile.read_bytes()
            # Validation chooses the actual 2000 target in its phase.
            task["level"]=2000.
            task["root"]["velocity_m_s"]=2000.
            engine.trial=lambda **kwargs:dict(success=True,status="success",velocity_m_s=2001.,
                jB=1.,continuation_state={"changed":True})
            engine.validate("bend")
            self.assertEqual(len(engine.payload["evaluations"]),2)
            self.assertEqual(engine.payload["validation"]["bend_2000"]["status"],"success")
            self.assertEqual(before,profile.read_bytes())

    def test_trial_timeout_is_retained_not_root(self):
        with tempfile.TemporaryDirectory() as d:
            engine,task=self.prepared(d,lambda density:500.)
            timeouts=[]
            def fail(**kwargs):
                timeouts.append(kwargs["timeout"])
                return dict(success=False,status="trial_timeout")
            engine.trial=fail
            engine.solve_target(task)
            self.assertTrue(all(0<t<=180 for t in timeouts))
            self.assertIsNone(task["root"])

    def test_all_completed_work_is_saved_before_target_callback(self):
        with tempfile.TemporaryDirectory() as d:
            engine,task=self.prepared(d,lambda density:500.)
            states=[]
            def completed(row,payload):
                states.append(targeted.load(engine.output)["targets"][row["id"]]["status"])
            engine.run_phase("bend",2,completed)
            self.assertEqual(states,["success"])

    def test_snapshot_is_byte_identical_and_original_not_modified(self):
        from _targeted_io import snapshot_inputs,read_inputs,verify_snapshot
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/"source";source.mkdir()
            import shutil
            for name in ("isothermal-domain.npy","isothermal-numerical.npy","isothermal-analytic.npy","isothermal-contour-repair.npy"):
                shutil.copyfile(OLD/name,source/name)
            expected={p.name:p.read_bytes() for p in source.iterdir()}
            destination=Path(d)/"new"
            snapshot=snapshot_inputs(destination,source)
            verify_snapshot(snapshot)
            for name,data in expected.items():
                self.assertEqual((source/name).read_bytes(),data)
                self.assertEqual((snapshot/name).read_bytes(),data)
            self.assertEqual(snapshot_inputs(destination,source),snapshot)

    def test_preflight_reports_missing_profiles_without_trials(self):
        import subprocess
        result=subprocess.run([sys.executable,str(WORK/"run_isothermal_numerical.py"),"--dry-run"],
                              cwd=ROOT,text=True,capture_output=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn("20 initial targets",result.stdout)
        self.assertIn("no automatic anchor rebuilding",result.stdout)

    def test_resume_repairs_parent_completion_before_children_saved(self):
        with tempfile.TemporaryDirectory() as d:
            engine,task=self.prepared(d,lambda density:500.+1000.*(density-4.09))
            engine.solve_target(task)
            self.assertEqual(task["status"],"needs_refinement")
            task["children"]=[]
            engine.save()
            resumed=self.engine(d,trial=lambda **kwargs:dict(success=False,status="trial_timeout"))
            resumed.run_phase("bend",1)
            self.assertEqual(len(resumed.payload["targets"][task["id"]]["children"]),2)

    def test_resume_summarizes_accuracy_check_without_repeating_it(self):
        with tempfile.TemporaryDirectory() as d:
            engine,task=self.prepared(d,lambda density:500.)
            engine.solve_target(task)
            task["phase"]="warm"
            engine.save()
            check=dict(task,id="check:warm_500")
            engine.shoot(check,task["root"],controls=engine.controls,purpose="accuracy_check")
            self.assertEqual(len(engine.payload["evaluations"]),2)
            resumed=self.engine(d,trial=lambda **kwargs:self.fail("Saved validation must not rerun"))
            resumed.validate("warm")
            self.assertEqual(len(resumed.payload["evaluations"]),2)
            self.assertEqual(resumed.payload["validation"]["warm_500"]["status"],"success")


if __name__ == "__main__":
    unittest.main()
