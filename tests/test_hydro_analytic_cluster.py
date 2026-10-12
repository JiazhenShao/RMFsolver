import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "new_paper_calculations" / "7Jun" / "run_hydro_analytic_contour.py"
SPEC = importlib.util.spec_from_file_location("run_hydro_analytic_contour", SCRIPT_PATH)
cluster = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cluster)


def test_lte_boundary_uses_aqstar_lte_quantity_and_failure():
    if cluster._boundary_quantity_name("LTE") != "a_interface_LTE":
        raise AssertionError("LTE boundary must use a_interface_LTE")
    if cluster._boundary_semantics("LTE") != "a_interface_LTE = 1":
        raise AssertionError("saved LTE boundary semantics must identify a_interface_LTE = 1")

    lte_failure = cluster.failure_values(
        "a_interface_LTE must satisfy 0 < a_interface_LTE < 1",
        "LTE",
    )
    if not lte_failure["endpoint_domain_failure"]:
        raise AssertionError("LTE a_interface_LTE failure should bracket the adaptive boundary")

    old_A_failure = cluster.failure_values(
        "aQstar_max must satisfy 0 < aQstar_max < 1",
        "LTE",
    )
    if old_A_failure["endpoint_domain_failure"]:
        raise AssertionError("LTE boundary must not be classified from the old A-domain error")


def test_extreme_endothermic_boundary_keeps_legacy_A_quantity():
    if cluster._boundary_quantity_name("extreme_endothermic") != "A_boundary":
        raise AssertionError("legacy boundary must use A_boundary")
    if cluster._boundary_semantics("extreme_endothermic") != "A_boundary = 1":
        raise AssertionError("saved legacy boundary semantics must identify A_boundary = 1")

    failure = cluster.failure_values(
        "A_boundary must satisfy 0 < A_boundary < 1",
        "extreme_endothermic",
    )
    if not failure["endpoint_domain_failure"]:
        raise AssertionError("legacy A failure should bracket the extreme-endothermic boundary")


def test_boundary_without_valid_probe_stays_nan(monkeypatch=None):
    original_calculate_point = cluster.calculate_point

    def always_fail(task):
        return {
            "success": False,
            "endpoint_domain_failure": False,
            "message": "synthetic non-endpoint failure",
            "T_MeV": task[0],
            "metastability": task[1],
            "aQstar_LTE": float("nan"),
            "A_extreme_endothermic": float("nan"),
        }

    cluster.calculate_point = always_fail
    try:
        row = cluster.solve_boundary_for_metastability(
            (
                0.1,
                1.0,
                3.0,
                180.0,
                0.0,
                0.0,
                "PNM",
                5000,
                0.1,
                100.0,
                3,
                1,
                "LTE",
            )
        )
    finally:
        cluster.calculate_point = original_calculate_point

    if row["success"]:
        raise AssertionError("boundary with no valid probes must not be marked successful")
    if row["status"] != "no_valid_probe":
        raise AssertionError(f"unexpected boundary status {row['status']!r}")
    if row["temperature_cut_MeV"] == row["temperature_cut_MeV"]:
        raise AssertionError("no-valid boundary must keep temperature_cut_MeV as NaN")


def test_runtime_api_validation_rejects_stale_phase_velocity():
    class OldPhaseVelocity:
        __file__ = "/fake/RMFsolver/phase_velocity.py"

        @staticmethod
        def analytic_velocity_bound(muB_N, T_N, B_one_forth):
            return None

    try:
        cluster._validate_phase_velocity_runtime(OldPhaseVelocity)
    except RuntimeError as exc:
        if "aQstar_max" not in str(exc):
            raise AssertionError(f"unexpected stale API error: {exc}") from exc
    else:
        raise AssertionError("runtime validation should reject stale analytic_velocity_bound")


def test_boundary_pass_validation_reports_hidden_worker_error():
    rows = [
        {
            "success": False,
            "status": "no_valid_probe",
            "metastability": 0.1,
            "evaluations": [
                {
                    "success": False,
                    "message": "analytic_velocity_bound() got an unexpected keyword argument 'aQstar_max'",
                }
            ],
        }
    ]
    try:
        cluster._validate_boundary_pass(rows, expected_count=1)
    except RuntimeError as exc:
        message = str(exc)
        if "unexpected keyword argument" not in message:
            raise AssertionError(f"worker failure was hidden: {message}") from exc
        if "no_valid_probe" not in message:
            raise AssertionError(f"boundary status was omitted: {message}") from exc
    else:
        raise AssertionError("invalid boundary pass should stop before velocity calculation")
