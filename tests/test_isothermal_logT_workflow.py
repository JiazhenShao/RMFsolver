"""Cheap orchestration tests: no live EOS or BVP calls."""
import copy
import importlib
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

WORK = Path(__file__).resolve().parents[1] / "new_paper_calculations/26-10-01/isothermal_numeric_logT"
sys.path.insert(0, str(WORK))


def module(name):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        return None


class GridTests(unittest.TestCase):
    def setUp(self):
        self.c = module("_logT_common")
        self.assertIsNotNone(self.c, "log-temperature workflow is not implemented")

    def test_exact_axes_and_target_counts(self):
        c = self.c
        np.testing.assert_allclose(c.TEMPERATURES, np.geomspace(.01, 20, 31))
        self.assertEqual(len(c.base_targets()), 217)
        self.assertEqual(sum(t["T_0minus"] == 20 for t in c.base_targets()), 7)
        self.assertEqual(len(c.anchor_ids()), 49)
        chains = c.block_chains()
        self.assertEqual(len(chains), 42)
        ids = [i for chain in chains for i in chain]
        self.assertEqual(len(ids), 168)
        self.assertEqual(len(set(ids) | c.anchor_ids()), 217)

    def test_density_step_scales_with_distance_from_coexistence(self):
        p = dict(nB_0minus_over_n0=3.001, velocity_m_s=75.)
        guess = self.c.propose_density([p], [], 3.001, 50., (3., 5.3))
        self.assertLess(guess, 3.001)
        self.assertGreater(guess, 3.)
        self.assertLess(abs(guess - 3.001), .001)

    def test_secant_uses_actual_numerical_bracket(self):
        points = [dict(nB_0minus_over_n0=3.001, velocity_m_s=50.),
                  dict(nB_0minus_over_n0=3.003, velocity_m_s=150.)]
        self.assertAlmostEqual(self.c.propose_density(points, [], 4., 100., (3., 5.3)), 3.002)

    def test_current_bounds_include_both_estimates(self):
        self.assertEqual(self.c.current_bounds([4., 8.]), (1., 32.))
        self.assertTrue(self.c.near_bound(1.0001, (1., 32.)))

    def test_target_identifiers_and_smoke_are_physical(self):
        tasks = self.c.base_targets(smoke=True)
        self.assertTrue(any(t["T_0minus"] == .01 and t["level"] == 50 for t in tasks))
        self.assertTrue(any(t["T_0minus"] == .01 and t["level"] == 5000 for t in tasks))
        self.assertTrue(any(t["T_0minus"] == 20 for t in tasks))
        self.assertTrue(all(.01 <= t["T_0minus"] <= 20 for t in tasks))


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.m = module("_logT_engine")
        self.c = module("_logT_common")
        self.assertIsNotNone(self.m, "log-temperature engine is not implemented")

    def engine(self, directory, **kwargs):
        inputs = dict(points=[], n0=1.e6, fingerprint="saved-domain", warm={})
        return self.m.ContourRun(directory, inputs, {"test": 1}, **kwargs)

    def task(self, engine, level=100., T=.01):
        task = self.c.target(level, T, "base")
        engine.payload["targets"][task["id"]] = task
        engine.payload["boundaries"][self.c.temperature_id(T)] = dict(status="success", lower=3., upper=5.3)
        return task

    def test_reservation_survives_restart_and_point_cap(self):
        with tempfile.TemporaryDirectory() as d:
            r = self.engine(d)
            t = self.task(r)
            p = self.c.point(.01, 3.001, 1.e6)
            self.assertIsNotNone(r.reserve(t, p))
            self.assertIsNotNone(r.reserve(t, p))
            self.assertIsNone(r.reserve(t, p))
            r2 = self.engine(d)
            self.assertEqual(len(r2.payload["evaluations"]), 2)
            self.assertTrue(all(e["status"] == "interrupted" for e in r2.payload["evaluations"]))
            self.assertIsNone(r2.reserve(r2.payload["targets"][t["id"]], p))

    def test_target_global_budget_and_validation_reserve(self):
        with tempfile.TemporaryDirectory() as d:
            r = self.engine(d, controls={"max_attempts": 18})
            t = self.task(r)
            for i in range(4):
                self.assertIsNotNone(r.reserve(t, self.c.point(.01, 3.001 + i*.001, 1.e6)))
                r.payload["evaluations"][-1].update(status="failed", runtime_s=1.)
            self.assertIsNone(r.reserve(t, self.c.point(.01, 3.1, 1.e6)))
            t2 = self.task(r, level=200.)
            self.assertIsNone(r.reserve(t2, self.c.point(.01, 3.2, 1.e6)))
            check = dict(t2, id="check:test", kind="validation")
            self.assertIsNotNone(r.reserve(check, self.c.point(.01, 3.2, 1.e6)))

    def test_exhausted_run_does_not_spend_time_on_analytical_proposals(self):
        with tempfile.TemporaryDirectory() as d:
            r=self.engine(d,controls={'max_attempts':14},
                proposal=lambda *args:self.fail('Proposal must not run with no numerical budget'))
            task=self.task(r)
            result=r.solve_target(task['id'])
            self.assertEqual(result['status'],'unresolved')
            self.assertIn('budget exhausted',result['message'])

    def test_resume_rejects_identity_change(self):
        with tempfile.TemporaryDirectory() as d:
            self.engine(d)
            with self.assertRaisesRegex(ValueError, "identity"):
                self.m.ContourRun(d, {"points": [], "n0": 1.e6}, {"test": 2})

    def test_resume_charges_interrupted_auxiliary_and_boundaries(self):
        with tempfile.TemporaryDirectory() as d:
            r = self.engine(d)
            t = self.task(r)
            t['aux_runtime_s'] = 60.
            r.payload['auxiliary'].append(dict(task_id=t['id'],status='running',reserved_seconds=60.))
            r.payload['boundaries'][self.c.temperature_id(.1)] = dict(status='running')
            r.save()
            resumed = self.engine(d)
            self.assertEqual(resumed.payload['auxiliary'][0]['status'],'interrupted')
            self.assertEqual(resumed.payload['boundaries'][self.c.temperature_id(.1)]['status'],'interrupted')
            self.assertEqual(resumed.spent(resumed.payload['targets'][t['id']]),60.)

    def test_smoke_profile_can_live_in_immutable_input_directory(self):
        with tempfile.TemporaryDirectory() as d:
            r = self.engine(Path(d)/'smoke')
            r.input_directory = Path(d)
            self.c.atomic_save(Path(d)/'inputs/profile.npy',{'s':np.array([0.,.9])})
            self.assertIsNotNone(r.profile({'profile_file':'inputs/profile.npy'}))

    def test_resume_releases_unlaunched_accuracy_placeholder_only(self):
        with tempfile.TemporaryDirectory() as d:
            r=self.engine(d)
            r.payload['validation']['check:unlaunched']={'status':'running'}
            r.save()
            resumed=self.engine(d)
            self.assertNotIn('check:unlaunched',resumed.payload['validation'])

    def test_resume_recovers_saved_accuracy_evaluation_without_retry(self):
        with tempfile.TemporaryDirectory() as d:
            r=self.engine(d)
            base=self.task(r)
            base.update(status='success',root={'velocity_m_s':100.})
            key='check:'+base['id']
            r.payload['validation'][key]={'status':'running','base_target':base['id']}
            check=dict(base,id=key,kind='validation')
            token=r.reserve(check,self.c.point(.01,3.001,1.e6))
            r.payload['evaluations'][token].update(status='success',result=dict(
                velocity_m_s=100.,jB=1.,current_bounds=(.1,10.)))
            r.save()
            resumed=self.engine(d)
            self.assertEqual(resumed.payload['validation'][key]['status'],'success')
            self.assertEqual(len(resumed.payload['evaluations']),1)

    def test_saves_before_progress_and_keeps_success(self):
        def trial(**kw):
            p = kw["point"]
            return dict(success=True, status="success", velocity_m_s=100., jB=1.,
                        continuation_state={"s": np.array([0., .999])}, validated=True)
        with tempfile.TemporaryDirectory() as d:
            observed = []
            r = self.engine(d, trial=trial, proposal=lambda *a: 3.002,
                evaluation_saved=lambda e: observed.append(self.c.load(Path(d)/"isothermal-logT.npy")["evaluations"][-1]["status"]))
            t = self.task(r)
            r.solve_target(t["id"])
            self.assertEqual(observed, ["success"])
            self.assertEqual(t["status"], "success")
            before = copy.deepcopy(r.payload["points"])
            r.solve_target(t["id"])
            self.assertEqual(len(r.payload["evaluations"]), 1)
            self.assertEqual(list(before), list(r.payload["points"]))

    def test_failed_work_is_not_contour_completion(self):
        with tempfile.TemporaryDirectory() as d:
            r = self.engine(d)
            t = self.task(r)
            t["status"] = "unresolved"
            r.finalize()
            self.assertTrue(r.payload["work_complete"])
            self.assertFalse(r.payload["contours_complete"])

    def test_refinement_is_one_midpoint_per_original_interval(self):
        with tempfile.TemporaryDirectory() as d:
            r = self.engine(d)
            for t in self.c.base_targets():
                r.payload["targets"][t["id"]] = dict(t, status="unresolved")
            r.schedule_refinements()
            count = len(r.payload["targets"])
            self.assertEqual(count, 217 + 210)
            r.schedule_refinements()
            self.assertEqual(len(r.payload["targets"]), count)

    def test_mocked_217_target_workflow_and_14_checks(self):
        def trial(**kwargs):
            physical = kwargs['point']
            velocity = (physical['nB_0minus_over_n0']-3.)*1.e5
            return dict(success=True,validated=True,status='success',velocity_m_s=velocity,
                        jB=physical['nB_0minus']*velocity/self.c.C)
        with tempfile.TemporaryDirectory() as directory:
            run = self.engine(directory,trial=trial,
                proposal=lambda task,*args: 3.+task['level']*1.e-5)
            for task in self.c.base_targets():
                run.payload['targets'][task['id']] = task
                run.payload['boundaries'][self.c.temperature_id(task['T_0minus'])] = dict(
                    status='success',lower=3.,upper=5.3)
            run.run_base(4)
            self.assertEqual(sum(t['status']=='success' for t in run.payload['targets'].values()),217)
            run.schedule_refinements()
            self.assertEqual(len(run.payload['targets']),217)
            for task in run.payload['targets'].values():
                if task['T_0minus'] in (.01,20.):
                    run.validate_target(task)
            run.finalize()
            self.assertTrue(run.payload['contours_complete'])
            self.assertEqual(len(run.payload['evaluations']),231)
            resumed = self.engine(directory,trial=lambda **kw:self.fail('Repeated exhausted work'))
            resumed.run_base(4)
            self.assertEqual(len(resumed.payload['evaluations']),231)


if __name__ == "__main__":
    unittest.main()
