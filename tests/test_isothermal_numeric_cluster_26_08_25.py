import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = (
    ROOT / "new_paper_calculations" / "26-08-25" / "isothermal_numeric"
)
RUNNER_PATH = MODULE_DIR / "run_isothermal_numerical.py"


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "isothermal_numeric_0825_runner", RUNNER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    private_module_names = ("_isothermal_domain", "_isothermal_execution")
    saved_private_modules = {
        name: sys.modules.pop(name, None) for name in private_module_names
    }
    sys.path.insert(0, str(MODULE_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(MODULE_DIR))
        for name in private_module_names:
            sys.modules.pop(name, None)
            if saved_private_modules[name] is not None:
                sys.modules[name] = saved_private_modules[name]
    return module


class IsothermalNumericRunnerTests(unittest.TestCase):
    def test_runner_is_numerical_only_and_has_portable_defaults(self):
        self.assertTrue(RUNNER_PATH.is_file())
        self.assertFalse((MODULE_DIR / "run_isothermal_analytic.py").exists())
        runner = load_runner()

        args = runner.build_parser().parse_args([])
        self.assertEqual(runner.REPO_ROOT, ROOT)
        self.assertEqual(args.domain, MODULE_DIR / "isothermal-domain.npy")
        self.assertEqual(
            args.analytic_seed,
            MODULE_DIR.parent / "isothermal_analytic" / "isothermal-analytic.npy",
        )
        self.assertEqual(args.output, MODULE_DIR / "isothermal-numerical.npy")
        self.assertEqual(args.tail_eps, 1.0e-3)

    def test_help_describes_the_numerical_stage(self):
        completed = subprocess.run(
            [sys.executable, str(RUNNER_PATH), "--help"],
            cwd=MODULE_DIR,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("exact numerical isothermal contour", completed.stdout)
        self.assertIn("--analytic-seed", completed.stdout)

    def test_main_reuses_domain_and_analytical_seed_then_runs_numerical_stage(self):
        runner = load_runner()
        domain_payload = {
            "payload_kind": "isothermal_domain",
            "ms_MeV": 0.0,
            "NM_type": "PNM",
            "upB": 5000,
        }
        analytic_payload = {"payload_kind": "isothermal_analytic"}
        numerical_payload = {
            "task_status": np.asarray([["success", "failure"]], dtype=object)
        }

        with (
            patch.object(
                runner.domain,
                "load_payload",
                side_effect=[domain_payload, analytic_payload],
            ) as load_payload,
            patch.object(runner.domain, "assert_domain_fingerprint"),
            patch.object(runner.domain, "validate_cluster_physics"),
            patch.object(
                runner.execution,
                "run_numerical_stage",
                return_value=numerical_payload,
            ) as run_stage,
        ):
            result = runner.main(["--workers", "7"])

        self.assertEqual(result, 0)
        self.assertEqual(load_payload.call_count, 2)
        self.assertEqual(
            load_payload.call_args_list[0].kwargs["expected_kind"],
            "isothermal_domain",
        )
        self.assertEqual(
            load_payload.call_args_list[1].kwargs["expected_kind"],
            "isothermal_analytic",
        )
        self.assertIs(run_stage.call_args.kwargs["domain_payload"], domain_payload)
        self.assertIs(
            run_stage.call_args.kwargs["analytic_payload"], analytic_payload
        )
        self.assertEqual(run_stage.call_args.kwargs["workers"], 7)

    def test_cell_core_calls_numerical_solver_with_automatic_composition(self):
        runner = load_runner()
        captured = {}

        def numerical_solver(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return {
                "success": True,
                "message": "ok",
                "a_0plus_source": "maximum",
                "a_0plus_max_status": "interior",
                "a_0plus_max": 0.5,
                "a_0plus_derived": 0.5,
                "jB": 2.0e-4,
                "u_0minus": 5.0e-5,
                "Pi": 20.0,
                "momentum_flux_ratio": 1.0e-7,
                "tail_residual_norm": 1.0e-7,
                "boundary_residuals": np.asarray([0.0, 0.0, 0.0]),
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

        task = {
            "ray_index": 0,
            "radial_index": 1,
            "T": 10.0,
            "nB_0minus": 4.0,
            "B_one_forth": 189.1565957288247,
            "ms": 0.0,
            "NM_type": "PNM",
            "tail_eps": 1.0e-3,
            "n_mesh": 300,
            "tol_bvp": 1.0e-4,
            "max_nodes": 50000,
            "kappa_factor": 1.0,
            "jB_lower_bound": 1.0e-12,
            "jB_upper_bound": 4.0e-4,
        }
        record = runner.execution.run_numerical_trial(
            task, 2.0e-4, "analytic", solver=numerical_solver
        )

        self.assertEqual(len(captured["args"]), 3)
        self.assertNotIn("a_0plus", captured["kwargs"])
        self.assertEqual(captured["kwargs"]["tail_eps"], 1.0e-3)
        self.assertEqual(captured["kwargs"]["NM_type"], "PNM")
        self.assertEqual(record["task_status"], "success")


if __name__ == "__main__":
    unittest.main()
