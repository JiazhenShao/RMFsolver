import importlib
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from RMFsolver import constants as const


ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT / "new_paper_calculations" / "26-08-19"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))


def sleep_then_return(seconds):
    time.sleep(float(seconds))
    return {"success": True}


class RecordingProgress:
    def __init__(self, events, **kwargs):
        self.events = events
        self.desc = kwargs["desc"]
        self.total = int(kwargs["total"])
        self.n = int(kwargs.get("initial", 0))
        events.append(("open", self.desc, self.total, self.n))

    def update(self, amount=1):
        self.n += int(amount)
        self.events.append(("update", self.desc, self.n))

    def close(self):
        self.events.append(("close", self.desc, self.n))


def recording_progress_factory(events):
    return lambda **kwargs: RecordingProgress(events, **kwargs)


class AxisAndUnitsTests(unittest.TestCase):
    def test_production_axes_and_physics_defaults(self):
        domain = importlib.import_module("_isothermal_domain")

        self.assertEqual(domain.DEFAULT_B_ONE_FORTH_MEV, 189.1565957288247)
        self.assertEqual(domain.DEFAULT_XI, -0.5)
        self.assertEqual(domain.DEFAULT_NM_TYPE, "PNM")
        self.assertEqual(domain.DEFAULT_MS_MEV, 0.0)
        self.assertEqual(domain.DEFAULT_UPB, 5000)
        self.assertEqual(domain.RUN_TAG, "isothermal-contour-26-08-19")

        temperature = domain.default_temperature_axis()
        composition = domain.default_a_0plus_target_axis()
        self.assertEqual(temperature.shape, (30,))
        self.assertEqual(composition.shape, (20,))
        self.assertEqual(temperature[0], 1.0e-2)
        self.assertEqual(temperature[-1], 120.0)
        self.assertEqual(composition[0], 0.01)
        self.assertEqual(composition[-1], 0.99)
        np.testing.assert_allclose(
            domain.phase_temperature_axis()[1:], temperature
        )
        self.assertEqual(domain.phase_temperature_axis()[0], 0.0)

    def test_proper_velocity_conversion_is_relativistic_and_unbounded_in_u(self):
        domain = importlib.import_module("_isothermal_domain")

        self.assertEqual(domain.proper_velocity_to_m_s(0.0), 0.0)
        for u in (1.0e-4, 1.0, 12.0):
            expected = domain.SPEED_OF_LIGHT_M_S * u / np.sqrt(1.0 + u * u)
            self.assertAlmostEqual(domain.proper_velocity_to_m_s(u), expected)
            self.assertAlmostEqual(
                domain.proper_velocity_to_m_s(-u), -expected
            )
        self.assertLess(
            domain.proper_velocity_to_m_s(12.0), domain.SPEED_OF_LIGHT_M_S
        )

    def test_atomic_payload_round_trip_and_contract_mismatch(self):
        domain = importlib.import_module("_isothermal_domain")
        payload = domain.base_payload(
            kind="test",
            temperature_axis=np.array([1.0, 2.0]),
            a_0plus_target_axis=np.array([0.2, 0.8]),
            B_one_forth=180.0,
            xi=-0.5,
            ms=0.0,
            NM_type="PNM",
            upB=5000,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "payload.npy"
            domain.atomic_save(path, payload)
            loaded = domain.load_payload(path, expected_kind="test")
            np.testing.assert_array_equal(
                loaded["temperature_axis_MeV"], np.array([1.0, 2.0])
            )
            loaded["schema_version"] += 1
            domain.atomic_save(path, loaded)
            with self.assertRaisesRegex(RuntimeError, "schema"):
                domain.load_payload(path, expected_kind="test")

    def test_process_pool_capability_handles_sandboxed_sysconf(self):
        domain = importlib.import_module("_isothermal_domain")
        with patch.object(os, "sysconf", side_effect=PermissionError("sandboxed")):
            self.assertFalse(domain.process_pool_available())


class DomainPhysicsTests(unittest.TestCase):
    N0 = 0.16 * const.MeV_fm**3
    B14 = 189.1565957288247

    def test_live_ceiling_closes_weighted_chemical_balance(self):
        domain = importlib.import_module("_isothermal_domain")
        ceiling = domain.ceiling_at_upstream(
            T=0.0,
            nB_0minus=3.5 * self.N0,
            B_one_forth=self.B14,
        )
        self.assertEqual(ceiling["status"], "interior")
        self.assertTrue(0.0 < ceiling["a_0plus_max"] < 1.0)
        self.assertLess(abs(ceiling["weighted_mu_residual_MeV"]), 1.0e-6)

    def test_live_phase_boundaries_are_ordered(self):
        domain = importlib.import_module("_isothermal_domain")
        lower = domain.solve_lower_phase_boundary(
            1.0, self.B14, previous_muB_0minus=1100.0
        )
        upper = domain.solve_upper_phase_boundary(
            1.0, self.B14, previous_state=(1640.0, 587.0)
        )
        self.assertEqual(lower["status"], "success")
        self.assertEqual(upper["status"], "success")
        self.assertLess(lower["nB_0minus"], upper["nB_0minus"])

    def test_inverse_ceiling_reproduces_target_fraction(self):
        domain = importlib.import_module("_isothermal_domain")
        lower = domain.solve_lower_phase_boundary(
            1.0, self.B14, previous_muB_0minus=1100.0
        )
        upper = domain.solve_upper_phase_boundary(
            1.0, self.B14, previous_state=(1640.0, 587.0)
        )
        solved = domain.solve_nB_0minus_for_a_0plus_max(
            T=1.0,
            a_0plus_target=0.5,
            lower_nB_0minus=lower["nB_0minus"],
            upper_nB_0minus=upper["nB_0minus"],
            lower_muB_0minus=lower["muB_0minus"],
            upper_muB_0minus=upper["muB_0minus"],
            B_one_forth=self.B14,
        )
        self.assertEqual(solved["status"], "success")
        self.assertLess(abs(solved["a_0plus_residual"]), 1.0e-8)
        self.assertLess(lower["nB_0minus"], solved["nB_0minus"])
        self.assertLess(solved["nB_0minus"], upper["nB_0minus"])

    def test_unordered_boundaries_produce_a_masked_row(self):
        domain = importlib.import_module("_isothermal_domain")
        row = domain.build_interior_row(
            T=20.0,
            a_0plus_target_axis=np.array([0.2, 0.8]),
            lower_boundary={"status": "success", "nB_0minus": 2.0},
            upper_boundary={"status": "success", "nB_0minus": 1.0},
            B_one_forth=self.B14,
        )
        self.assertEqual(row["row_status"], "no_allowed_band")
        self.assertTrue(np.all(row["cell_status"] == "no_allowed_band"))
        self.assertTrue(np.all(np.isnan(row["nB_0minus"])))

    def test_domain_payload_has_shared_curvilinear_grid(self):
        domain = importlib.import_module("_isothermal_domain")

        def lower(T, *_args, **_kwargs):
            return {
                "status": "success",
                "T_MeV": float(T),
                "nB_0minus": 2.0,
                "muB_0minus": 1000.0,
            }

        def upper(T, *_args, **_kwargs):
            return {
                "status": "success",
                "T_MeV": float(T),
                "nB_0minus": 4.0,
                "muB_0minus": 1400.0,
                "muB_0plus": 1100.0,
                "muK_0plus": 300.0,
            }

        def point(**kwargs):
            target = float(kwargs["a_0plus_target"])
            return {
                "status": "success",
                "nB_0minus": 2.0 + 2.0 * target,
                "muB_0minus": 1000.0 + 400.0 * target,
                "P_0minus": 10.0 + target,
                "a_0plus_max": target,
                "a_0plus_residual": 0.0,
            }

        with (
            patch.object(domain, "solve_lower_phase_boundary", side_effect=lower),
            patch.object(domain, "solve_upper_phase_boundary", side_effect=upper),
            patch.object(
                domain, "solve_nB_0minus_for_a_0plus_max", side_effect=point
            ),
        ):
            payload = domain.build_domain_payload(
                temperature_axis=np.array([1.0, 20.0]),
                a_0plus_target_axis=np.array([0.2, 0.8]),
                B_one_forth=self.B14,
                xi=-0.5,
                ms=0.0,
                NM_type="PNM",
                upB=5000,
                workers=1,
            )
        self.assertEqual(payload["nB_0minus_grid"].shape, (2, 2))
        self.assertEqual(payload["phase_temperature_axis_MeV"].shape, (3,))
        self.assertTrue(payload["run_complete"])
        self.assertEqual(
            payload["domain_fingerprint"], domain.compute_domain_fingerprint(payload)
        )
        corrupted = dict(payload)
        corrupted["nB_0minus_grid"] = payload["nB_0minus_grid"].copy()
        corrupted["nB_0minus_grid"][0, 0] += 1.0
        with self.assertRaisesRegex(RuntimeError, "fingerprint"):
            domain.assert_domain_fingerprint(corrupted)
        np.testing.assert_allclose(
            payload["a_0plus_max_grid"], [[0.2, 0.8], [0.2, 0.8]]
        )

    def test_domain_checkpoints_and_advances_after_every_boundary_and_grid_point(self):
        domain = importlib.import_module("_isothermal_domain")
        events = []

        def lower(T, *_args, **_kwargs):
            return {
                "status": "success",
                "T_MeV": float(T),
                "nB_0minus": 2.0,
                "muB_0minus": 1000.0,
            }

        def upper(T, *_args, **_kwargs):
            return {
                "status": "success",
                "T_MeV": float(T),
                "nB_0minus": 4.0,
                "muB_0minus": 1400.0,
                "muB_0plus": 1100.0,
                "muK_0plus": 300.0,
            }

        def point(**kwargs):
            target = float(kwargs["a_0plus_target"])
            return {
                "status": "success",
                "nB_0minus": 2.0 + 2.0 * target,
                "muB_0minus": 1000.0 + 400.0 * target,
                "P_0minus": 10.0 + target,
                "a_0plus_max": target,
                "a_0plus_residual": 0.0,
            }

        def save(_path, _payload):
            events.append(("save",))
            return Path(_path)

        with (
            patch.object(domain, "solve_lower_phase_boundary", side_effect=lower),
            patch.object(domain, "solve_upper_phase_boundary", side_effect=upper),
            patch.object(
                domain, "solve_nB_0minus_for_a_0plus_max", side_effect=point
            ),
            patch.object(domain, "atomic_save", side_effect=save),
        ):
            payload = domain.build_domain_payload(
                temperature_axis=np.array([1.0, 20.0]),
                a_0plus_target_axis=np.array([0.2, 0.8]),
                B_one_forth=self.B14,
                xi=-0.5,
                ms=0.0,
                NM_type="PNM",
                upB=5000,
                workers=1,
                output_path=Path("domain.npy"),
                progress_factory=recording_progress_factory(events),
            )
        opens = [event for event in events if event[0] == "open"]
        self.assertEqual(
            [(event[1], event[2]) for event in opens],
            [
                ("Stable neutron matter boundary", 3),
                ("a(0+)=1 boundary", 3),
                ("Domain grid", 4),
            ],
        )
        self.assertEqual(len([event for event in events if event[0] == "update"]), 10)
        self.assertEqual(len([event for event in events if event[0] == "save"]), 10)
        self.assertTrue(payload["run_complete"])

    def test_domain_grid_uses_spawned_point_workers_when_parallel(self):
        domain = importlib.import_module("_isothermal_domain")

        def lower(T, *_args, **_kwargs):
            return {
                "status": "success",
                "T_MeV": float(T),
                "nB_0minus": 2.0,
                "muB_0minus": 1000.0,
            }

        def upper(T, *_args, **_kwargs):
            return {
                "status": "success",
                "T_MeV": float(T),
                "nB_0minus": 4.0,
                "muB_0minus": 1400.0,
                "muB_0plus": 1100.0,
                "muK_0plus": 300.0,
            }

        def point(**kwargs):
            target = float(kwargs["a_0plus_target"])
            return {
                "status": "success",
                "nB_0minus": 2.0 + target,
                "muB_0minus": 1000.0 + target,
                "P_0minus": 10.0,
                "a_0plus_max": target,
                "a_0plus_residual": 0.0,
            }

        def spawned(task):
            return domain._build_cell_task(dict(task))

        with (
            patch.object(domain, "solve_lower_phase_boundary", side_effect=lower),
            patch.object(domain, "solve_upper_phase_boundary", side_effect=upper),
            patch.object(
                domain, "solve_nB_0minus_for_a_0plus_max", side_effect=point
            ),
            patch.object(
                domain, "_build_cell_in_spawned_process", side_effect=spawned
            ) as spawn_runner,
        ):
            payload = domain.build_domain_payload(
                temperature_axis=np.array([1.0, 20.0]),
                a_0plus_target_axis=np.array([0.2, 0.8]),
                B_one_forth=self.B14,
                xi=-0.5,
                ms=0.0,
                NM_type="PNM",
                upB=5000,
                workers=4,
            )
        self.assertEqual(spawn_runner.call_count, 4)
        self.assertTrue(payload["run_complete"])

    def test_domain_resume_skips_terminal_boundaries_and_cells(self):
        domain = importlib.import_module("_isothermal_domain")

        def lower(T, *_args, **_kwargs):
            return {
                "status": "success",
                "T_MeV": float(T),
                "nB_0minus": 2.0,
                "muB_0minus": 1000.0,
            }

        def upper(T, *_args, **_kwargs):
            return {
                "status": "success",
                "T_MeV": float(T),
                "nB_0minus": 4.0,
                "muB_0minus": 1400.0,
                "muB_0plus": 1100.0,
                "muK_0plus": 300.0,
            }

        def point(**kwargs):
            target = float(kwargs["a_0plus_target"])
            return {
                "status": "success",
                "nB_0minus": 2.0 + target,
                "muB_0minus": 1000.0 + target,
                "P_0minus": 10.0,
                "a_0plus_max": target,
                "a_0plus_residual": 0.0,
            }

        build_kwargs = {
            "temperature_axis": np.array([1.0, 20.0]),
            "a_0plus_target_axis": np.array([0.2, 0.8]),
            "B_one_forth": self.B14,
            "xi": -0.5,
            "ms": 0.0,
            "NM_type": "PNM",
            "upB": 5000,
            "workers": 1,
        }
        with (
            patch.object(domain, "solve_lower_phase_boundary", side_effect=lower),
            patch.object(domain, "solve_upper_phase_boundary", side_effect=upper),
            patch.object(
                domain, "solve_nB_0minus_for_a_0plus_max", side_effect=point
            ),
        ):
            partial = domain.build_domain_payload(**build_kwargs)
        partial["run_complete"] = False
        partial["domain_fingerprint"] = None
        partial["row_status"][1] = "pending"
        partial["cell_status"][1, 1] = "pending"
        partial["cell_records"][1, 1] = None
        for key in (
            "nB_0minus_grid",
            "muB_0minus_grid",
            "P_0minus_grid",
            "a_0plus_max_grid",
            "a_0plus_residual_grid",
        ):
            partial[key][1, 1] = np.nan

        events = []

        def save(_path, _payload):
            events.append(("save",))
            return Path(_path)

        with (
            patch.object(domain, "solve_lower_phase_boundary") as lower_solver,
            patch.object(domain, "solve_upper_phase_boundary") as upper_solver,
            patch.object(
                domain, "solve_nB_0minus_for_a_0plus_max", side_effect=point
            ) as point_solver,
            patch.object(domain, "atomic_save", side_effect=save),
        ):
            resumed = domain.build_domain_payload(
                **build_kwargs,
                output_path=Path("domain.npy"),
                progress_factory=recording_progress_factory(events),
                resume_payload=partial,
            )
        lower_solver.assert_not_called()
        upper_solver.assert_not_called()
        self.assertEqual(point_solver.call_count, 1)
        self.assertEqual(len([event for event in events if event[0] == "save"]), 1)
        updates = [event for event in events if event[0] == "update"]
        self.assertEqual(updates, [("update", "Domain grid", 4)])
        self.assertTrue(resumed["run_complete"])
        domain.assert_domain_fingerprint(resumed)


class AnalyticalStageTests(unittest.TestCase):
    def _task(self):
        return {
            "temperature_index": 1,
            "composition_index": 2,
            "T_0minus": 20.0,
            "nB_0minus": 3.5,
            "B_one_forth": 189.1565957288247,
            "a_0plus_target": 0.5,
            "xi": -0.5,
            "ms": 0.0,
            "NM_type": "PNM",
            "upB": 5000,
        }

    def test_analytic_call_omits_a_0plus_and_converts_proper_velocity(self):
        execution = importlib.import_module("_isothermal_execution")
        captured = {}

        def solver(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return {
                "success": True,
                "status": "moving_front",
                "front_exists": True,
                "a_0plus_source": "maximum",
                "a_0plus_max": 0.5,
                "a_0plus_max_status": "interior",
                "u_0minus": 0.25,
                "jB": 0.875,
                "delta_muB": -1.0,
            }

        record = execution.run_analytic_task(self._task(), solver=solver)
        self.assertEqual(len(captured["args"]), 3)
        self.assertNotIn("a_0plus", captured["kwargs"])
        self.assertEqual(captured["kwargs"]["xi"], -0.5)
        self.assertEqual(captured["kwargs"]["ms"], 0.0)
        self.assertEqual(captured["kwargs"]["NM_type"], "PNM")
        self.assertEqual(captured["kwargs"]["upB"], 5000)
        self.assertEqual(record["task_status"], "success")
        expected = 299_792_458.0 * 0.25 / np.sqrt(1.0 + 0.25**2)
        self.assertAlmostEqual(record["velocity_m_s"], expected)

    def test_analytic_validity_statuses_are_masked_without_relabeling(self):
        execution = importlib.import_module("_isothermal_execution")
        for status in (
            "slow_front_approximation_invalid",
            "momentum_flux_ratio_above_tolerance",
            "composition_ceiling_saturated",
        ):
            with self.subTest(status=status):
                record = execution.run_analytic_task(
                    self._task(),
                    solver=lambda *_args, **_kwargs: {
                        "success": False,
                        "status": status,
                        "front_exists": False,
                        "a_0plus_source": "maximum",
                        "a_0plus_max": 0.5,
                        "u_0minus": np.nan,
                        "jB": np.nan,
                    },
                )
                self.assertEqual(record["task_status"], status)
                self.assertTrue(np.isnan(record["velocity_m_s"]))

    def test_analytic_timeout_is_a_structured_terminal_record(self):
        execution = importlib.import_module("_isothermal_execution")

        def runner(_task, _timeout_s):
            raise execution.BVPTrialTimeoutError("timed out")

        record = execution.run_analytic_task_with_timeout(
            self._task(), timeout_s=0.1, runner=runner
        )
        self.assertEqual(record["task_status"], "timeout")
        self.assertTrue(record["child_terminated"])
        self.assertTrue(np.isnan(record["velocity_m_s"]))

    def test_analytic_rejects_composition_or_phase_disagreement(self):
        execution = importlib.import_module("_isothermal_execution")
        mismatched = execution.run_analytic_task(
            self._task(),
            solver=lambda *_args, **_kwargs: {
                "success": True,
                "status": "moving_front",
                "front_exists": True,
                "a_0plus_source": "maximum",
                "a_0plus_max": 0.6,
                "u_0minus": 0.1,
                "jB": 0.35,
            },
        )
        self.assertEqual(mismatched["task_status"], "composition_mismatch")
        stable = execution.run_analytic_task(
            self._task(),
            solver=lambda *_args, **_kwargs: {
                "success": True,
                "status": "stable_neutron_matter",
                "front_exists": False,
                "a_0plus_source": "maximum",
                "a_0plus_max": np.nan,
                "u_0minus": 0.0,
                "jB": 0.0,
            },
        )
        self.assertEqual(stable["task_status"], "domain_solver_disagreement")

    def test_analytic_diagnostics_are_checkpointed_as_scalar_grids(self):
        domain = importlib.import_module("_isothermal_domain")
        execution = importlib.import_module("_isothermal_execution")
        domain_payload = domain.base_payload(
            kind="isothermal_domain",
            temperature_axis=np.array([10.0]),
            a_0plus_target_axis=np.array([0.5]),
            B_one_forth=189.1565957288247,
            xi=-0.5,
            ms=0.0,
            NM_type="PNM",
            upB=5000,
        )
        domain_payload.update(
            {
                "nB_0minus_grid": np.array([[3.5]]),
                "a_0plus_max_grid": np.array([[0.5]]),
            }
        )
        payload = execution._method_payload(domain_payload, "isothermal_analytic")
        execution._store_record(
            payload,
            {
                "temperature_index": 0,
                "composition_index": 0,
                "task_status": "success",
                "u_0minus": 0.1,
                "velocity_m_s": 100.0,
                "jB": 0.35,
                "momentum_flux_ratio": 1.0e-6,
                "gamma_minus_1": 0.005,
                "slow_front_consistent": True,
                "a_0plus_max_status": "interior",
                "a_0plus_max_residual": 1.0e-12,
                "delta_muB": -2.0,
                "analytic_velocity_residual": 3.0e-13,
                "momentum_flux_inf_residual": 4.0e-13,
                "momentum_flux_0plus_residual": 5.0e-13,
                "composition_residual": 6.0e-13,
            },
        )
        self.assertEqual(payload["a_0plus_max_status_grid"][0, 0], "interior")
        self.assertTrue(payload["slow_front_consistent_grid"][0, 0])
        self.assertEqual(payload["delta_muB_grid"][0, 0], -2.0)
        self.assertEqual(payload["composition_residual_grid"][0, 0], 6.0e-13)

    def test_analytic_resume_rejects_a_different_domain_fingerprint(self):
        domain = importlib.import_module("_isothermal_domain")
        execution = importlib.import_module("_isothermal_execution")
        domain_payload = domain.base_payload(
            kind="isothermal_domain",
            temperature_axis=np.array([10.0]),
            a_0plus_target_axis=np.array([0.5]),
            B_one_forth=189.1565957288247,
            xi=-0.5,
            ms=0.0,
            NM_type="PNM",
            upB=5000,
        )
        domain_payload.update(
            {
                "domain_fingerprint": "new-domain",
                "cell_status": np.array([["success"]], dtype=object),
                "nB_0minus_grid": np.array([[3.5]]),
                "a_0plus_max_grid": np.array([[0.5]]),
            }
        )
        stale = execution._method_payload(domain_payload, "isothermal_analytic")
        stale["domain_fingerprint"] = "old-domain"
        stale["analytical_controls"] = {
            "timeout_s": execution.DEFAULT_ANALYTIC_TIMEOUT_S
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "analytic.npy"
            domain.atomic_save(output, stale)
            with self.assertRaisesRegex(RuntimeError, "fingerprint"):
                execution.run_analytic_stage(
                    domain_payload=domain_payload,
                    output_path=output,
                    workers=1,
                    resume=True,
                )

    def test_analytic_progress_follows_each_point_checkpoint(self):
        domain = importlib.import_module("_isothermal_domain")
        execution = importlib.import_module("_isothermal_execution")
        domain_payload = domain.base_payload(
            kind="isothermal_domain",
            temperature_axis=np.array([10.0]),
            a_0plus_target_axis=np.array([0.2, 0.8]),
            B_one_forth=189.1565957288247,
            xi=-0.5,
            ms=0.0,
            NM_type="PNM",
            upB=5000,
        )
        domain_payload.update(
            {
                "cell_status": np.full((1, 2), "success", dtype=object),
                "nB_0minus_grid": np.full((1, 2), 4.0),
                "a_0plus_max_grid": np.array([[0.2, 0.8]]),
            }
        )
        events = []

        def solve(task, **_kwargs):
            return {
                "success": True,
                "task_status": "success",
                "temperature_index": task["temperature_index"],
                "composition_index": task["composition_index"],
                "jB": 1.0e-4,
                "u_0minus": 2.5e-5,
                "velocity_m_s": 100.0,
            }

        def save(_path, _payload):
            events.append(("save",))
            return Path(_path)

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(execution, "run_analytic_task_with_timeout", side_effect=solve),
            patch.object(domain, "atomic_save", side_effect=save),
        ):
            execution.run_analytic_stage(
                domain_payload=domain_payload,
                output_path=Path(directory) / "analytic.npy",
                workers=1,
                progress_factory=recording_progress_factory(events),
            )
        point_events = [event[0] for event in events if event[0] in {"save", "update"}]
        self.assertEqual(point_events[:4], ["save", "update", "save", "update"])
        opens = [event for event in events if event[0] == "open"]
        self.assertEqual(opens[0][1:3], ("Analytical contour scan", 2))


class NumericalStageTests(unittest.TestCase):
    def _task(self):
        return {
            "temperature_index": 0,
            "composition_index": 1,
            "T": 10.0,
            "nB_0minus": 4.0,
            "B_one_forth": 189.1565957288247,
            "a_0plus_target": 0.5,
            "ms": 0.0,
            "NM_type": "PNM",
            "tail_eps": 1.0e-8,
            "n_mesh": 300,
            "tol_bvp": 1.0e-4,
            "max_nodes": 50000,
            "kappa_factor": 1.0,
            "jB_lower_bound": 1.0e-12,
            "jB_upper_bound": 4.0e-4,
        }

    def test_numerical_call_omits_a_0plus_and_uses_exact_bvp_defaults(self):
        execution = importlib.import_module("_isothermal_execution")
        captured = {}

        def solver(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return {
                "success": True,
                "message": "ok",
                "a_0plus_source": "maximum",
                "a_0plus_max_status": "interior",
                "a_0plus_max": 0.5,
                "a_0plus": 0.5,
                "a_0plus_derived": 0.5,
                "jB": 2.0e-4,
                "u_0minus": 5.0e-5,
                "Pi": 20.0,
                "momentum_flux_ratio": 1.0e-7,
                "tail_residual_norm": 1.0e-7,
                "boundary_residuals": np.array([0.0, 0.0, 0.0]),
                "rate_model": "exact_nonleptonic",
                "diffusion_model": "local_muB_fixed_T",
                "composition_definition": "nK_over_local_nB",
                "current_definition": "u_nK_minus_D_K_dnK_dx",
                "coordinate": "BVP: s in [0, 1-tail_eps]",
                "compact_scale": 2.0,
                "s_end": 0.999,
                "bvp_status": 0,
                "bvp_niter": 2,
                "bvp_nodes": 300,
            }

        record = execution.run_numerical_trial(
            self._task(), 2.0e-4, "analytic", solver=solver
        )
        self.assertEqual(len(captured["args"]), 3)
        self.assertNotIn("a_0plus", captured["kwargs"])
        self.assertEqual(captured["kwargs"]["ms"], 0.0)
        self.assertEqual(captured["kwargs"]["NM_type"], "PNM")
        self.assertEqual(captured["kwargs"]["tail_eps"], 1.0e-8)
        self.assertEqual(captured["kwargs"]["n_mesh"], 300)
        self.assertEqual(captured["kwargs"]["tol_bvp"], 1.0e-4)
        self.assertEqual(captured["kwargs"]["max_nodes"], 50000)
        self.assertEqual(record["task_status"], "success")

    def test_numerical_validation_rejects_nonfinite_bvp_residual(self):
        execution = importlib.import_module("_isothermal_execution")

        def solver(*_args, **_kwargs):
            return {
                "success": True,
                "message": "bad residual",
                "a_0plus_source": "maximum",
                "a_0plus_max_status": "interior",
                "a_0plus_max": 0.5,
                "a_0plus_derived": 0.5,
                "jB": 2.0e-4,
                "u_0minus": 5.0e-5,
                "Pi": 20.0,
                "momentum_flux_ratio": 1.0e-7,
                "tail_residual_norm": np.nan,
                "boundary_residuals": np.zeros(3),
                "rate_model": "exact_nonleptonic",
                "diffusion_model": "local_muB_fixed_T",
                "composition_definition": "nK_over_local_nB",
                "current_definition": "u_nK_minus_D_K_dnK_dx",
                "coordinate": "BVP compact coordinate",
                "compact_scale": 2.0,
                "s_end": 0.999,
                "bvp_status": 0,
                "bvp_niter": 2,
                "bvp_nodes": 300,
            }

        record = execution.run_numerical_trial(
            self._task(), 2.0e-4, "analytic", solver=solver
        )
        self.assertEqual(record["task_status"], "invalid_bvp_diagnostics")

    def test_candidate_order_prefers_same_temperature_history(self):
        execution = importlib.import_module("_isothermal_execution")
        candidates = execution.build_jB_candidates(
            nB_0minus=4.0,
            previous_jB=1.0e-4,
            earlier_jB=8.0e-5,
            analytic_jB=1.2e-4,
            nearby_shell_jB=(1.1e-4, 1.3e-4),
            lower_bound=1.0e-12,
            upper_bound=4.0e-4,
        )
        self.assertEqual(candidates[0][1], "previous_composition_shell")
        self.assertEqual(candidates[1][1], "log_shell_extrapolation")
        self.assertEqual(candidates[2][1], "analytic_current_cell")
        self.assertIn("log_extrapolation_multiplier_2", [source for _, source in candidates])
        values = np.asarray([value for value, _source in candidates])
        self.assertTrue(np.all(values > 1.0e-12))
        self.assertTrue(np.all(values < 4.0e-4))
        for left, right in zip(values[:-1], values[1:]):
            self.assertGreaterEqual(abs(left - right) / max(abs(left), abs(right)), 0.03)

    def test_hard_timeout_terminates_child(self):
        execution = importlib.import_module("_isothermal_execution")
        with self.assertRaises(execution.BVPTrialTimeoutError):
            execution.run_module_call_with_hard_timeout(
                "tests.test_isothermal_contour_cluster_26_08_19",
                "sleep_then_return",
                {"seconds": 2.0},
                timeout_s=0.1,
            )

    def test_cell_retries_candidates_until_success(self):
        execution = importlib.import_module("_isothermal_execution")
        calls = []

        def runner(task, jB_guess, source, timeout_s):
            calls.append((jB_guess, source, timeout_s))
            if len(calls) == 1:
                return {
                    "temperature_index": task["temperature_index"],
                    "composition_index": task["composition_index"],
                    "task_status": "solver_failure",
                    "success": False,
                    "message": "first guess failed",
                }
            return {
                "temperature_index": task["temperature_index"],
                "composition_index": task["composition_index"],
                "task_status": "success",
                "success": True,
                "jB": jB_guess,
                "u_0minus": jB_guess / task["nB_0minus"],
                "velocity_m_s": 100.0,
            }

        record = execution.solve_numerical_cell(
            self._task(),
            [(1.0e-4, "first"), (2.0e-4, "second")],
            trial_timeout_s=2.0,
            cell_timeout_s=5.0,
            trial_runner=runner,
        )
        self.assertEqual(record["task_status"], "success")
        self.assertEqual(len(record["attempts"]), 2)
        self.assertEqual(record["jB_guess_source"], "second")

    def test_stage_advances_in_composition_shells_and_checkpoints(self):
        domain = importlib.import_module("_isothermal_domain")
        execution = importlib.import_module("_isothermal_execution")
        domain_payload = domain.base_payload(
            kind="isothermal_domain",
            temperature_axis=np.array([1.0, 20.0]),
            a_0plus_target_axis=np.array([0.2, 0.8]),
            B_one_forth=189.1565957288247,
            xi=-0.5,
            ms=0.0,
            NM_type="PNM",
            upB=5000,
        )
        domain_payload.update(
            {
                "run_complete": True,
                "cell_status": np.full((2, 2), "success", dtype=object),
                "nB_0minus_grid": np.full((2, 2), 4.0),
                "a_0plus_max_grid": np.array([[0.2, 0.8], [0.2, 0.8]]),
            }
        )
        analytic_payload = execution._method_payload(
            domain_payload, "isothermal_analytic"
        )
        analytic_payload["jB_grid"][:] = 1.0e-4
        analytic_payload["task_status"][:] = "success"
        calls = []

        def solve(task, candidates, **_kwargs):
            calls.append((task["composition_index"], task["temperature_index"]))
            jB = 1.0e-4 * (task["composition_index"] + 1)
            return {
                "success": True,
                "task_status": "success",
                "temperature_index": task["temperature_index"],
                "composition_index": task["composition_index"],
                "jB": jB,
                "u_0minus": jB / task["nB_0minus"],
                "velocity_m_s": 100.0,
                "attempts": [],
            }

        with tempfile.TemporaryDirectory() as directory, patch.object(
            execution, "solve_numerical_cell", side_effect=solve
        ):
            payload = execution.run_numerical_stage(
                domain_payload=domain_payload,
                analytic_payload=analytic_payload,
                output_path=Path(directory) / "numerical.npy",
                workers=1,
            )
        self.assertEqual([item[0] for item in calls], [0, 0, 1, 1])
        self.assertTrue(np.all(payload["task_status"] == "success"))
        self.assertTrue(payload["run_complete"])

    def test_stage_resume_skips_terminal_cells(self):
        domain = importlib.import_module("_isothermal_domain")
        execution = importlib.import_module("_isothermal_execution")
        domain_payload = domain.base_payload(
            kind="isothermal_domain",
            temperature_axis=np.array([1.0]),
            a_0plus_target_axis=np.array([0.2, 0.8]),
            B_one_forth=189.1565957288247,
            xi=-0.5,
            ms=0.0,
            NM_type="PNM",
            upB=5000,
        )
        domain_payload.update(
            {
                "cell_status": np.full((1, 2), "success", dtype=object),
                "nB_0minus_grid": np.full((1, 2), 4.0),
                "a_0plus_max_grid": np.array([[0.2, 0.8]]),
            }
        )
        analytic = execution._method_payload(domain_payload, "isothermal_analytic")
        analytic["jB_grid"][:] = 1.0e-4
        analytic["task_status"][:] = "success"
        calls = []

        def solve(task, _candidates, **_kwargs):
            calls.append(task["composition_index"])
            return {
                "success": True,
                "task_status": "success",
                "temperature_index": 0,
                "composition_index": task["composition_index"],
                "jB": 1.0e-4,
                "u_0minus": 2.5e-5,
                "velocity_m_s": 100.0,
                "attempts": [],
            }

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "numerical.npy"
            partial = execution._method_payload(domain_payload, "isothermal_numerical")
            partial["task_status"][0, 0] = "success"
            partial["jB_grid"][0, 0] = 1.0e-4
            partial["numerical_controls"] = {
                "trial_timeout_s": execution.DEFAULT_TRIAL_TIMEOUT_S,
                "cell_timeout_s": execution.DEFAULT_CELL_TIMEOUT_S,
                "tail_eps": execution.DEFAULT_TAIL_EPS,
                "n_mesh": execution.DEFAULT_N_MESH,
                "tol_bvp": execution.DEFAULT_BVP_TOL,
                "max_nodes": execution.DEFAULT_MAX_NODES,
                "kappa_factor": execution.DEFAULT_KAPPA_FACTOR,
                "jB_lower_bound": execution.DEFAULT_JB_LOWER_BOUND,
                "jB_upper_factor": execution.DEFAULT_JB_UPPER_FACTOR,
            }
            domain.atomic_save(output, partial)
            with patch.object(execution, "solve_numerical_cell", side_effect=solve):
                result = execution.run_numerical_stage(
                    domain_payload=domain_payload,
                    analytic_payload=analytic,
                    output_path=output,
                    workers=1,
                    resume=True,
                )
        self.assertEqual(calls, [1])
        self.assertTrue(result["run_complete"])

    def test_numerical_resume_rejects_changed_controls(self):
        domain = importlib.import_module("_isothermal_domain")
        execution = importlib.import_module("_isothermal_execution")
        domain_payload = domain.base_payload(
            kind="isothermal_domain",
            temperature_axis=np.array([1.0]),
            a_0plus_target_axis=np.array([0.5]),
            B_one_forth=189.1565957288247,
            xi=-0.5,
            ms=0.0,
            NM_type="PNM",
            upB=5000,
        )
        domain_payload.update(
            {
                "cell_status": np.array([["success"]], dtype=object),
                "nB_0minus_grid": np.array([[4.0]]),
                "a_0plus_max_grid": np.array([[0.5]]),
            }
        )
        analytic = execution._method_payload(domain_payload, "isothermal_analytic")
        stale = execution._method_payload(domain_payload, "isothermal_numerical")
        stale["numerical_controls"] = {"cell_timeout_s": 1.0}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "numerical.npy"
            domain.atomic_save(output, stale)
            with self.assertRaisesRegex(RuntimeError, "controls"):
                execution.run_numerical_stage(
                    domain_payload=domain_payload,
                    analytic_payload=analytic,
                    output_path=output,
                    workers=1,
                    resume=True,
                )

    def test_numerical_parallel_results_checkpoint_before_each_progress_update(self):
        domain = importlib.import_module("_isothermal_domain")
        execution = importlib.import_module("_isothermal_execution")
        domain_payload = domain.base_payload(
            kind="isothermal_domain",
            temperature_axis=np.array([1.0, 20.0]),
            a_0plus_target_axis=np.array([0.5]),
            B_one_forth=189.1565957288247,
            xi=-0.5,
            ms=0.0,
            NM_type="PNM",
            upB=5000,
        )
        domain_payload.update(
            {
                "cell_status": np.full((2, 1), "success", dtype=object),
                "nB_0minus_grid": np.full((2, 1), 4.0),
                "a_0plus_max_grid": np.full((2, 1), 0.5),
            }
        )
        analytic = execution._method_payload(domain_payload, "isothermal_analytic")
        analytic["jB_grid"][:] = 1.0e-4
        analytic["task_status"][:] = "success"
        events = []

        def solve(task, _candidates, **_kwargs):
            return {
                "success": True,
                "task_status": "success",
                "temperature_index": task["temperature_index"],
                "composition_index": task["composition_index"],
                "jB": 1.0e-4,
                "u_0minus": 2.5e-5,
                "velocity_m_s": 100.0,
                "attempts": [],
            }

        def save(_path, _payload):
            events.append(("save",))
            return Path(_path)

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(execution, "solve_numerical_cell", side_effect=solve),
            patch.object(domain, "atomic_save", side_effect=save),
        ):
            execution.run_numerical_stage(
                domain_payload=domain_payload,
                analytic_payload=analytic,
                output_path=Path(directory) / "numerical.npy",
                workers=2,
                progress_factory=recording_progress_factory(events),
            )
        point_events = [event[0] for event in events if event[0] in {"save", "update"}]
        self.assertIn(["save", "update", "save", "update"], [point_events[i:i+4] for i in range(len(point_events)-3)])
        opens = [event for event in events if event[0] == "open"]
        self.assertEqual(opens[0][1:3], ("Numerical contour scan", 2))


class PlotAndDriverTests(unittest.TestCase):
    def _payloads(self):
        domain = importlib.import_module("_isothermal_domain")
        execution = importlib.import_module("_isothermal_execution")
        domain_payload = domain.base_payload(
            kind="isothermal_domain",
            temperature_axis=np.array([10.0, 20.0]),
            a_0plus_target_axis=np.array([0.2, 0.8]),
            B_one_forth=189.1565957288247,
            xi=-0.5,
            ms=0.0,
            NM_type="PNM",
            upB=5000,
        )
        domain_payload.update(
            {
                "phase_temperature_axis_MeV": np.array([0.0, 10.0, 20.0]),
                "lower_phase_nB_0minus": np.array([2.0, 2.1, 2.2])
                * const.NuclearDensity_nucleons_MeV3,
                "upper_phase_nB_0minus": np.array([5.0, 4.9, 4.8])
                * const.NuclearDensity_nucleons_MeV3,
                "cell_status": np.full((2, 2), "success", dtype=object),
                "nB_0minus_grid": np.array([[2.5, 4.5], [2.6, 4.4]])
                * const.NuclearDensity_nucleons_MeV3,
                "a_0plus_max_grid": np.array([[0.2, 0.8], [0.2, 0.8]]),
            }
        )
        analytic = execution._method_payload(domain_payload, "isothermal_analytic")
        numerical = execution._method_payload(domain_payload, "isothermal_numerical")
        for payload in (analytic, numerical):
            payload["task_status"][:] = "success"
            payload["velocity_m_s_grid"][:] = np.array([[50.0, 100.0], [200.0, 500.0]])
        analytic["task_status"][0, 1] = "momentum_flux_ratio_above_tolerance"
        analytic["velocity_m_s_grid"][0, 1] = np.inf
        return domain_payload, analytic, numerical

    def test_plot_mask_keeps_only_finite_successful_cells(self):
        plotting = importlib.import_module("plot_isothermal_contours")
        _domain, analytic, _numerical = self._payloads()
        velocity = plotting.masked_velocity(analytic)
        self.assertTrue(velocity.mask[0, 1])
        self.assertFalse(velocity.mask[1, 1])
        self.assertFalse(np.any(np.isinf(velocity.compressed())))

    def test_plot_shades_both_forbidden_regions(self):
        plotting = importlib.import_module("plot_isothermal_contours")
        domain_payload, analytic, numerical = self._payloads()
        with tempfile.TemporaryDirectory() as directory:
            figure, axes = plotting.plot_comparison(
                domain_payload,
                analytic,
                numerical,
                output_path=Path(directory) / "figure.png",
            )
            self.assertTrue((Path(directory) / "figure.png").exists())
        labels = {collection.get_label() for collection in axes[0].collections}
        self.assertIn("Stable PNM", labels)
        self.assertIn("a(0+) >= 1", labels)
        import matplotlib.pyplot as plt

        plt.close(figure)

    def test_all_stage_driver_orders_subprocesses_and_preserves_physics(self):
        driver = importlib.import_module("run_isothermal_all")
        args = driver.build_parser().parse_args(
            [
                "--workers",
                "7",
                "--B-one-forth",
                "190.0",
                "--xi",
                "-0.25",
                "--output-dir",
                "/tmp/isothermal-test",
                "--smoke",
            ]
        )
        commands = driver.build_stage_commands(args)
        self.assertEqual(
            [Path(command[1]).name for command in commands],
            [
                "prepare_isothermal_domain.py",
                "run_isothermal_analytic.py",
                "run_isothermal_numerical.py",
                "plot_isothermal_contours.py",
            ],
        )
        self.assertIn("190.0", commands[0])
        self.assertIn("-0.25", commands[0])
        self.assertIn("--smoke", commands[0])
        self.assertIn("7", commands[1])
        self.assertIn("7", commands[2])

    def test_all_stage_driver_rejects_inconsistent_numerical_upB(self):
        driver = importlib.import_module("run_isothermal_all")
        args = driver.build_parser().parse_args(["--upB", "4000"])
        with self.assertRaisesRegex(ValueError, "upB=5000"):
            driver.build_stage_commands(args)

    def test_all_stage_resume_reuses_the_existing_domain(self):
        driver = importlib.import_module("run_isothermal_all")
        args = driver.build_parser().parse_args(["--resume"])
        commands = driver.build_stage_commands(args)
        self.assertIn("--resume", commands[0])
        self.assertIn("--resume", commands[1])
        self.assertIn("--resume", commands[2])

    def test_phase_segments_do_not_bridge_invalid_rows(self):
        plotting = importlib.import_module("plot_isothermal_contours")
        segments = plotting._contiguous_valid_slices(
            np.array([True, True, False, True, False, True, True])
        )
        self.assertEqual(
            [(segment.start, segment.stop) for segment in segments],
            [(0, 2), (3, 4), (5, 7)],
        )


if __name__ == "__main__":
    unittest.main()
